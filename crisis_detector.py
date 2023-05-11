from typing import Iterator, List, Optional, Sized, Tuple, Iterable, Any
from warnings import warn
import os
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
import torchmetrics
import lightning as pl
from lightning.pytorch import seed_everything
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler

from embedder import TextEmbedder
from data_tools import load_data, SeriesDataset, get_all_data, get_data_with_dates
from training_tools import split_dataset, train_model, test_model, fold_dataset
from aggregator import EmbeddingAggregator, MeanBackbone

torch.set_float32_matmul_precision('high')

def add_embeddings(days_df: pd.DataFrame, text_df: pd.DataFrame, embeddings: List[torch.Tensor] | torch.Tensor, aggregator: EmbeddingAggregator) -> pd.DataFrame:
    if type(embeddings) == list:
        embeddings = torch.cat(embeddings, dim=0)
    # embeddings = embeddings.numpy()
    sections = np.cumsum(text_df.groupby(['group', 'Data wydania']).count()['text']).tolist()[:-1]
    embeddings = torch.vsplit(embeddings, sections)
    if aggregator.model.sample_size:
        day_embeddings = aggregator(pad_sequence(embeddings, batch_first=True)).numpy()
    else:
        day_embeddings = np.concatenate([aggregator(t.unsqueeze(0)).numpy() for t in embeddings], axis=0)
    # day_embeddings = np.stack([np.mean(t, axis=0) for t in np.vsplit(embeddings, sections)], axis=0)
    embedding_df = text_df[['group', 'Data wydania']].drop_duplicates().reset_index(drop=True)
    embedding_df['embedding'] = day_embeddings.tolist()
    days_df = days_df.join(embedding_df.set_index(['group', 'Data wydania']), ['group', 'Data wydania'], how='left')
    days_df.loc[:, 'embedding'].iloc[days_df['embedding'].isna()] = pd.Series(np.zeros((days_df['embedding'].isna().sum(), 768)).tolist())
    return days_df

def create_dataset(df: pd.DataFrame, sequence_len: int = 30) -> Tuple[TensorDataset, torch.Tensor]:
    numeric_cols = ['brak', 'pozytywny', 'neutralny', 'negatywny', 'suma']
    X = torch.tensor(df[numeric_cols].values, dtype=torch.long)
    y = torch.tensor(df['label'], dtype=torch.long)
    embeddings = torch.tensor(df['embedding'], dtype=torch.float32)
    groups = df['group']

    new_features = []
    for i in groups.unique():
        X_i = X[groups == i]
        X_diff = torch.ones_like(X_i)
        X_diff[1:] = X_i[1:] / X_i[:-1]
        X_diff = torch.pow(X_diff, 2)
        X_diff = (X_diff - 1.) / (X_diff + 1.)
        zeros = X_i == 0
        X_diff[zeros & X_diff.isnan()] = 0.
        X_diff.nan_to_num_(1.)
        X_ratio = X_i[:, :-1] / X_i[:, -1:]
        X_ratio.nan_to_num_(0.)
        X_scaled = torch.tensor(StandardScaler().fit_transform(X_i))
        new_features.append(torch.cat((X_scaled, X_diff, X_ratio), dim=1))
    
    X = torch.cat((torch.cat(new_features, dim=0), embeddings), dim=1).to(torch.float32)

    X_seq, y_seq, groups_seq = [], [], []
    for i in groups.unique():
        X_i = X[groups == i]
        y_i = y[groups == i]
        for j in range(X_i.shape[0] - sequence_len + 1):
            X_seq.append(X_i[j:j+sequence_len])
            y_seq.append(y_i[j+sequence_len-1])
            groups_seq.append(i)
    X_seq = torch.stack(X_seq)
    y_seq = torch.stack(y_seq)
    groups_seq = torch.tensor(groups_seq)

    return TensorDataset(X_seq, y_seq), groups_seq
    
class ShiftMetric(torchmetrics.Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, crisis_threshold: int = 5, absolute: bool = True, average: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = crisis_threshold
        self.absolute = absolute
        self.average = average
        self.add_state('shifts', default=[], dist_reduce_fx='cat')
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = preds.long(), target.long()
        diffs = (torch.diff(target, prepend=torch.tensor([0], dtype=target.dtype, device=target.device)) == 1).nonzero().squeeze(1)
        assert diffs.shape[0] == 1
        crisis_start = diffs[0]
        pred_starts = (torch.diff(
                preds,
                prepend=torch.tensor([0], dtype=preds.dtype, device=preds.device),
                append=torch.tensor([1] * self.threshold, dtype=preds.dtype, device=preds.device)
            ) == 1).nonzero().squeeze(1)
        for pred_start in pred_starts:
            if preds[pred_start:pred_start + self.threshold].all():
                shift = pred_start - crisis_start
                if self.absolute:
                    shift = shift.abs()
                self.shifts.append(shift)
                return
        raise RuntimeError('I broke Cauchy\'s theorem.')
    
    def compute(self) -> torch.Tensor:
        shifts = torch.tensor(self.shifts, device=self.device)
        if self.average:
            return shifts.float().mean()
        else:
            return shifts

class Shift2Metric(torchmetrics.Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, crisis_threshold: int = 5, average: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = crisis_threshold
        self.average = average
        self.add_state('shifts', default=[], dist_reduce_fx='cat')
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = preds.long(), target.long()
        diffs = (torch.diff(target, prepend=torch.tensor([0], dtype=target.dtype, device=target.device)) == 1).nonzero().squeeze(1)
        assert diffs.shape[0] == 1
        crisis_start = diffs[0]
        pred_starts = (torch.diff(
                preds,
                prepend=torch.tensor([0], dtype=preds.dtype, device=preds.device),
                append=torch.tensor([1] * self.threshold, dtype=preds.dtype, device=preds.device)
            ) == 1).nonzero().squeeze(1)
        for pred_start in pred_starts:
            if preds[pred_start:pred_start + self.threshold].all():
                shift = torch.max(pred_start - crisis_start, torch.tensor(0, dtype=torch.long, device=self.device))
                self.shifts.append(shift)
                return
        raise RuntimeError('I broke Cauchy\'s theorem.')
    
    def compute(self) -> torch.Tensor:
        shifts = torch.tensor(self.shifts, device=self.device)
        if self.average:
            return shifts.float().mean()
        else:
            return shifts

class TopicSampler(Sampler[List[int]]):
    def __init__(self, data_source: Dataset) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        labels = self.data_source[:][1]
        self.topic_idx = (torch.diff(
                labels,
                prepend=torch.tensor([1], dtype=labels.dtype, device=labels.device),
                append=torch.tensor([0], dtype=labels.dtype, device=labels.device)
            ) == -1).nonzero().squeeze(1)
    
    def __iter__(self) -> Iterator[List[int]]:
        for i in range(self.topic_idx.shape[0] - 1):
            yield list(range(self.topic_idx[i], self.topic_idx[i+1]))
    
    def __len__(self) -> int:
        return self.topic_idx.shape[0] - 1

class MyClassifier(pl.LightningModule):
    def __init__(
            self,
            input_dim: int = 782,
            lr: float = 0.001,
            weight_decay: float = 0.01,
            class_ratios: torch.Tensor | None = None,
            input_limit: slice | None = None,
            print_test_samples: bool = False
    ) -> None:
        super().__init__()
        self.input_limit = input_limit
        if self.input_limit is not None:
            self.input_dim = len(range(*self.input_limit.indices(input_dim)))
        else:
            self.input_dim = input_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.print_test_samples = print_test_samples
        self.class_weights = .5 / class_ratios if class_ratios else None

        self.loss_fn = nn.CrossEntropyLoss(self.class_weights)
        self.f1 = torchmetrics.F1Score('multiclass', num_classes=2, average=None)
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=2, average='micro')
        self.prec = torchmetrics.Precision('multiclass', num_classes=2, average=None)
        self.rec = torchmetrics.Recall('multiclass', num_classes=2, average=None)
        self.shift = ShiftMetric()
        self.shift2 = Shift2Metric()
    
    def forward(self, x):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        if type(batch) == tuple or type(batch) == list:
            X, y = batch
        elif type(batch) == dict:
            X, y = batch, batch['label']
        else:
            raise TypeError(f'Invalid batch type {type(batch)}. Expected tuple, list or dict.')
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if type(batch) == tuple or type(batch) == list:
            X, y = batch
        elif type(batch) == dict:
            X, y = batch, batch['label']
        else:
            raise TypeError(f'Invalid batch type {type(batch)}. Expected tuple, list or dict.')
        y_pred = self(X)
        self.acc.update(torch.argmax(y_pred, -1), y)
        self.f1.update(torch.argmax(y_pred, -1), y)
        self.log('val_loss', self.loss_fn(y_pred, y), on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        metrics = {
            'val_acc': self.acc.compute(),
            'val_f1': self.f1.compute().mean()
        }
        self.log_dict(metrics)
        self.acc.reset()
        self.f1.reset()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if type(batch) == tuple or type(batch) == list:
            X, y = batch
        elif type(batch) == dict:
            X, y = batch, batch['label']
        else:
            raise TypeError(f'Invalid batch type {type(batch)}. Expected tuple, list or dict.')
        y_pred = self(X)
        if self.print_test_samples:
            print(torch.argmax(y_pred, -1) - y)
        self.acc.update(torch.argmax(y_pred, -1), y)
        self.f1.update(torch.argmax(y_pred, -1), y)
        self.prec.update(torch.argmax(y_pred, -1), y)
        self.rec.update(torch.argmax(y_pred, -1), y)
        self.shift.update(torch.argmax(y_pred, -1), y)
        self.shift2.update(torch.argmax(y_pred, -1), y)
        self.log('test_loss', self.loss_fn(y_pred, y), on_epoch=True)
    
    def on_test_epoch_end(self) -> None:
        metrics = {
            'test_acc': self.acc.compute()
        }
        metrics['test_f1_neg'], metrics['test_f1_pos'] = self.f1.compute()
        metrics['test_precision_neg'], metrics['test_precision_pos'] = self.prec.compute()
        metrics['test_recall_neg'], metrics['test_recall_pos'] = self.rec.compute()
        shift = self.shift.compute().float()
        metrics['test_shift'], metrics['test_shift_std'] = shift.mean(), shift.std()
        shift = self.shift2.compute().float()
        metrics['test_shift2'], metrics['test_shift2_std'] = shift.mean(), shift.std()
        self.log_dict(metrics)
        self.acc.reset()
        self.f1.reset()
        self.prec.reset()
        self.rec.reset()
        self.shift.reset()
        self.shift2.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, 'min', .5, 10),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

class MyLSTM(MyClassifier):
    def __init__(
            self,
            hidden_dim: int = 128,
            **kwargs
    ) -> None:
        kwargs.setdefault('lr', 0.002)
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, 384),
                nn.Dropout(0.125),
                nn.Tanh(),
                nn.Linear(384, self.hidden_dim),
                nn.Dropout(0.125),
                nn.Tanh(),
                nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
            ),
            nn.Sequential(
                nn.Dropout(0.125),
                nn.Linear(self.hidden_dim, 2),
                nn.Softmax(dim=-1)
            )
        ])
    
    def forward(self, x):
        if self.input_limit is not None:
            x = x[..., self.input_limit]
        x, _ = self.nets[0](x)
        x = self.nets[1](x[:,-1,:])
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

class MyTransformer(MyClassifier):
    def __init__(
            self,
            hidden_dim: int = 128,
            n_heads: int = 8,
            transformer_layers: int = 6,
            **kwargs
    ) -> None:
        kwargs.setdefault('lr', 1e-5)
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.transformer_layers = transformer_layers

        self.input_net = nn.Sequential(
            nn.Linear(self.input_dim, 384),
            nn.Dropout(.1),
            nn.Tanh(),
            nn.Linear(384, 256),
            nn.Dropout(.1),
            nn.Tanh(),
            nn.Linear(256, self.hidden_dim),
            nn.Dropout(.1),
            nn.Tanh(),
        )
        self.positional_encoding = PositionalEncoding(self.hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.hidden_dim, self.n_heads, activation=nn.functional.leaky_relu, batch_first=True),
            self.transformer_layers
        )
        self.output_net = nn.Sequential(
            nn.Dropout(.1),
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        if self.input_limit is not None:
            x = x[..., self.input_limit]
        x = self.input_net(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)[:, -1]
        x = self.output_net(x)
        return x

def cross_validate(
        model_class: type,
        model_params: dict,
        ds: Dataset,
        groups: torch.Tensor,
        use_weights: bool = False,
        n_splits: int = 10,
        precision: str = 'bf16-mixed',
        batch_size: int = 512,
        max_epochs: int = -1,
        max_time = None,
        num_workers: int = 10,
        deterministic: bool = False
) -> pd.DataFrame:
    folds = fold_dataset(ds, groups, n_splits)
    stats = []
    for train_ds, test_ds, val_ds in tqdm(folds):
        if use_weights:
            class_ratio = train_ds[:][1].unique(return_counts=True)[1] / len(train_ds)
            weight = torch.pow(class_ratio * class_ratio.shape[0], -1)
            model_params += (weight,)
        model = model_class(**model_params)

        trainer = train_model(model, train_ds, val_ds, precision, batch_size, max_epochs, max_time, num_workers, False, deterministic)

        test_dl = DataLoader(test_ds, batch_sampler=TopicSampler(test_ds), num_workers=num_workers, pin_memory=True)
        test_results = trainer.test(dataloaders=test_dl, ckpt_path='best', verbose=False)[0]
        stats.append(test_results)
    stats = pd.DataFrame(stats)
    print(stats)
    print('Means:')
    print(stats.mean(axis=0))
    print('Standard deviation:')
    print(stats.std(axis=0))
    return stats

def get_shifts(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    sequence_idx = (torch.diff(y_true, prepend=torch.tensor([1], device=y_true.device), append=torch.tensor([0], device=y_true.device)) == -1).nonzero().squeeze()
    crisis_idx = (torch.diff(y_true, prepend=torch.tensor([0], device=y_true.device)) == 1).nonzero().squeeze()
    pred_start = torch.diff(y_pred, prepend=torch.tensor([0], device=y_true.device)) == 1
    shifts = torch.zeros_like(crisis_idx)
    if len(crisis_idx.shape) == 0:
        return torch.tensor([], dtype=torch.long, device=y_true.device)
    for i in range(crisis_idx.shape[0]):
        if y_pred[crisis_idx[i]]:
            search_interval = -1
        else:
            search_interval = 1
        while not pred_start[crisis_idx[i] + shifts[i]]:
            shifts[i] += search_interval
            if crisis_idx[i] + shifts[i] < sequence_idx[i] or crisis_idx[i] + shifts[i] >= sequence_idx[i+1]:
                break
    return shifts

def find_mistakes(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    sequence_idx = (torch.diff(y_true, prepend=torch.tensor([1]), append=torch.tensor([0])) == -1).nonzero().squeeze()
    y_true = [y_true[sequence_idx[i]:sequence_idx[i+1]] for i in range(len(sequence_idx)-1)]
    y_pred = [y_pred[sequence_idx[i]:sequence_idx[i+1]] for i in range(len(sequence_idx)-1)]
    crisis_len = torch.stack([t.unique(return_counts=True)[1] for t in y_true], dim=0).max(dim=0)[0]
    for i in range(len(y_true)):
        lens = y_true[i].unique(return_counts=True)[1]
        y_true[i] = nn.functional.pad(y_true[i], tuple(crisis_len - lens), value=-1)
        y_pred[i] = nn.functional.pad(y_pred[i], tuple(crisis_len - lens), value=-1)
    y_true = torch.stack(y_true, dim=0)
    y_pred = torch.stack(y_pred, dim=0)
    return y_pred - y_true

@torch.no_grad()
def test_shift(model: pl.LightningModule, test_ds: Dataset, verbose: bool = False) -> Tuple[float, float]:
    X, y = test_ds[:]
    y_pred = torch.argmax(model(X), dim=-1)
    if verbose:
        print(find_mistakes(y, y_pred))
    shifts = get_shifts(y, y_pred).abs().float()
    return shifts.mean().item(), shifts.std().item()

def save_shift(model: pl.LightningModule, test_ds: Dataset, crisis_names: Iterable[str], filename: str) -> None:
    X, y = test_ds[:]
    y_pred = torch.argmax(model(X), dim=-1)
    shifts = get_shifts(y, y_pred)
    d = {}
    for i, name in enumerate(crisis_names):
        d[name] = shifts[i].item()
    with open(filename, 'w') as f:
        json.dump(d, f)

def main():
    deterministic = True
    end_to_end = False
    text_samples = 50

    if deterministic:
        seed_everything(42, workers=True)

    DAYS_DF_PATH = 'saved_objects/days_df.feather'
    POSTS_DF_PATH = 'saved_objects/posts_df' + str(text_samples) + '.feather'
    EMBEDDINGS_PATH = 'saved_objects/embeddings' + str(text_samples) + '.pt'

    if end_to_end or not (os.path.isfile(DAYS_DF_PATH) and os.path.isfile(POSTS_DF_PATH)):

        data = get_data_with_dates(get_all_data())

        days_df, text_df = load_data(data['path'].to_list(), data['crisis_start'].to_list(), text_samples, True)
        days_df.to_feather(DAYS_DF_PATH)
        text_df.to_feather(POSTS_DF_PATH)

        if deterministic:
            seed_everything(42, workers=True)
    else:
        days_df = pd.read_feather(DAYS_DF_PATH)
        text_df = pd.read_feather(POSTS_DF_PATH)

    if end_to_end or not os.path.isfile(EMBEDDINGS_PATH):
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        ds = SeriesDataset(text_df['text'])
        collate_fn = lambda x: tokenizer(x, truncation=True, padding=True, max_length=256, return_tensors='pt')
        dl = DataLoader(ds, 64, num_workers=10, collate_fn=collate_fn, pin_memory=True)
        model = TextEmbedder.load_from_checkpoint('saved_objects/finetuned-xlm-roberta-base.ckpt')
        # model = TextEmbedder('sdadas/polish-distilroberta')
        trainer = pl.Trainer(precision='bf16-mixed', logger=False, deterministic=deterministic)
        embeddings = trainer.predict(model, dl)
        embeddings = torch.cat(embeddings, dim=0)

        with open(EMBEDDINGS_PATH, 'wb') as f:
            torch.save(embeddings, f)
        
        if deterministic:
            seed_everything(42, workers=True)
    else:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = torch.load(f)

    aggregator = EmbeddingAggregator(MeanBackbone())
    days_df = add_embeddings(days_df, text_df, embeddings, aggregator)
    ds, groups = create_dataset(days_df)

    # train_ds, test_ds, val_ds = split_dataset(ds, groups)
    # model = MyTransformer(print_test_samples=True)
    # trainer = train_model(model, train_ds, val_ds, precision='bf16-mixed', deterministic=deterministic)
    # test_dl = DataLoader(test_ds, batch_sampler=TopicSampler(test_ds), num_workers=10, pin_memory=True)
    # trainer.test(dataloaders=test_dl, ckpt_path='best', verbose=True)

    cross_validate(MyTransformer, {}, ds, groups, n_splits=5, precision='bf16-mixed', deterministic=deterministic)

if __name__ == '__main__':
    main()
