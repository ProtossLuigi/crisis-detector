import math
from typing import Any, Optional, Tuple, Iterable
import os
import numpy as np
import pandas as pd
from random import sample
from warnings import warn
from tqdm import tqdm

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torchmetrics
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_tools import get_data_with_dates, get_verified_data, SeriesDataset, SimpleDataset
from training_tools import init_trainer, predefined_split, split_dataset, fold_dataset, train_model
from embedder import TextEmbedder

torch.set_float32_matmul_precision('high')

class EmbeddingAggregator(pl.LightningModule):
    def __init__(
            self,
            sample_size: int = 0,
            embedding_dim: int = 768,
            lr: float = 1e-3,
            weight_decay: float = 0.01,
            train_dataloader_len: int | None = None, 
            warmup_proportion: float = .1, 
            max_epochs: int | None = None,
            class_ratios: torch.Tensor | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.sample_size = sample_size
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_dataloader_len = train_dataloader_len
        self.warmup_proportion = warmup_proportion
        self.max_epochs = max_epochs
        
        self.init_loss_fn(class_ratios)

        self.loss_fn = nn.CrossEntropyLoss(self.class_weights)
        self.f1 = torchmetrics.F1Score('multiclass', num_classes=2, average=None)
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=2, average='micro')
        self.prec = torchmetrics.Precision('multiclass', num_classes=2, average=None)
        self.rec = torchmetrics.Recall('multiclass', num_classes=2, average=None)
        
        self.classifier = nn.Sequential(
            nn.Dropout(.1),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 2)
        )
    
    def init_loss_fn(self, class_ratios: torch.Tensor | None = None) -> None:
        self.class_weights = .5 / class_ratios if class_ratios is not None else None
        self.loss_fn = nn.CrossEntropyLoss(self.class_weights)
    
    def forward(self, x) -> Any:
        raise NotImplementedError()
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, y = batch
        y_pred = self.classifier(self(X))
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss.item(), on_epoch=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        X, y = batch
        y_pred = self.classifier(self(X))
        self.acc.update(torch.argmax(y_pred, -1), y)
        self.f1.update(torch.argmax(y_pred, -1), y)
        self.log('val_loss', self.loss_fn(y_pred, y), on_epoch=True)
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        metrics = {
            'val_acc_macro': self.acc.compute(),
            'val_f1_macro': self.f1.compute().mean()
        }
        self.log_dict(metrics)
        self.acc.reset()
        self.f1.reset()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        X, y = batch
        y_pred = self.classifier(self(X))
        self.acc.update(torch.argmax(y_pred, -1), y)
        self.f1.update(torch.argmax(y_pred, -1), y)
        self.prec.update(torch.argmax(y_pred, -1), y)
        self.rec.update(torch.argmax(y_pred, -1), y)
        self.log('test_loss', self.loss_fn(y_pred, y), on_epoch=True)
    
    @torch.no_grad()
    def on_test_epoch_end(self) -> None:
        metrics = {
            'test_acc': self.acc.compute()
        }
        f1 = self.f1.compute()
        metrics['test_f1_0'], metrics['test_f1_1'], metrics['test_f1_macro'] = f1[0], f1[1], f1.mean(dim=0)
        prec = self.prec.compute()
        metrics['test_precision_0'], metrics['test_precision_1'] = prec[0], prec[1]
        rec = self.rec.compute()
        metrics['test_recall_0'], metrics['test_recall_1'] = rec[0], rec[1]
        self.log_dict(metrics)
        self.acc.reset()
        self.f1.reset()
        self.prec.reset()
        self.rec.reset()
    
    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if type(batch) == list:
            batch = torch.cat(batch)
        return self(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.warmup_proportion is not None and self.train_dataloader_len is not None:
            num_training_steps = self.max_epochs * self.train_dataloader_len
            num_warmup_steps = int(self.warmup_proportion * num_training_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer

class MeanAggregator(EmbeddingAggregator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return torch.mean(x, dim=1)

class ConvAggregator(EmbeddingAggregator):
    def __init__(self, sample_size: int = 50, *args, **kwargs) -> None:
        super().__init__(sample_size, *args, **kwargs)
        if self.sample_size != 50:
            raise NotImplementedError()

        self.net = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.embedding_dim, 10, 2),
            nn.Tanh(),
            nn.Conv1d(self.embedding_dim, self.embedding_dim, 5, 2),
            nn.Tanh(),
            nn.Conv1d(self.embedding_dim, self.embedding_dim, 9),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor):
        return self.net(x.view(-1, self.embedding_dim, self.sample_size)).squeeze()
   
class RecurrentAggregator(EmbeddingAggregator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.net = nn.GRU(self.embedding_dim, self.embedding_dim, num_layers=2, batch_first=True, dropout=.1)
    
    def forward(self, x: torch.Tensor):
        x, _ = self.net(x)
        return x[:, -1, :]

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

class TransformerAggregator(EmbeddingAggregator):
    def __init__(self, n_heads: int = 8, transformer_layers: int = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads
        self.transformer_layers = transformer_layers

        # self.positional_encoding = PositionalEncoding(self.embedding_dim, max_len=self.sample_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.embedding_dim, self.n_heads, activation=nn.functional.relu, batch_first=True),
            self.transformer_layers
        )

        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor):
        # x = self.positional_encoding(x)
        return self.transformer(x).mean(dim=1)

class MaskedAggregator(EmbeddingAggregator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.sample_size < 1:
            raise ValueError('Sample size must be greater than 0.')
        
        self.mlp1 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(.1),
            nn.PReLU(),
            # nn.Linear(self.embedding_dim, self.embedding_dim),
            # nn.Dropout(),
            # nn.PReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.sample_size, self.sample_size),
            nn.Dropout(.1),
            nn.RReLU(),
            # nn.Linear(self.sample_size, self.sample_size),
            # nn.Dropout(),
            # nn.PReLU(),
            nn.Linear(self.sample_size, self.sample_size),
            nn.Softmax()
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh()
        )
    
    def forward(self, x) -> Any:
        embedding_mask = self.mlp1(x.flatten(0, 1)).view(x.shape)
        sample_mask = self.mlp2(x.mT.flatten(0, 1)).view(x.shape[0], x.shape[2], x.shape[1]).mT
        return self.mlp3((embedding_mask * sample_mask * x).mean(dim=1))

def extract_data(
        filename: str,
        crisis_start: pd.Timestamp,
        window_size: int | Tuple[int, int] = 30,
        drop_invalid: bool = False
) -> pd.DataFrame | None:
    src_df = pd.read_feather(filename)

    if type(window_size) == int:
        window_size = (window_size, window_size)
    window = (crisis_start - pd.Timedelta(days=window_size[0]), crisis_start + pd.Timedelta(days=window_size[1]))

    src_df = src_df[(window[0] <= src_df['Data wydania']) & (src_df['Data wydania'] < window[1])]

    if src_df['Kryzys'].hasnans:
        if src_df['Kryzys'].nunique(dropna=False) != 2:
            if drop_invalid:
                return None
            else:
                warn(f'Invalid Kryzys column values in {filename}.')
        labels = ~src_df['Kryzys'].isna()
    else:
        src_df['Kryzys'] = src_df['Kryzys'].apply(lambda x: x[:3])
        if src_df['Kryzys'].nunique(dropna=False) != 2:
            if drop_invalid:
                return None
            else:
                warn(f'Invalid Kryzys column values in {filename}.')
        labels = src_df['Kryzys'] != 'NIE'

    text = src_df.apply(lambda x: " . ".join([str(x['Tytuł publikacji']), str(x['Lead']), str(x['Kontekst publikacji'])]), axis=1)
    text_df = pd.DataFrame({'text': text, 'label': labels, 'date': src_df['Data wydania']})
    
    return text_df

def load_data(filenames: Iterable[str], crisis_dates: Iterable[pd.Timestamp], drop_invalid: bool = False) -> pd.DataFrame:
    assert len(filenames) == len(crisis_dates)
    dfs = []
    for i, (fname, date) in enumerate(tqdm(zip(filenames, crisis_dates), total=len(filenames))):
        try:
            df = extract_data(fname, date, drop_invalid=drop_invalid)
        except KeyError:
            warn(f'Invalid columns in {fname}')
            df = None
        if df is None:
            continue
        df['group'] = i
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def create_dataset(df: pd.DataFrame, embeddings: torch.Tensor, threshold: float = 0., max_samples: int = 0, batched: bool = True, padding: bool = False, balance_classes: bool = False):
    assert (batched and max_samples) or not padding
    labels = torch.tensor((df[['group', 'date', 'label']].groupby(['group', 'date']).mean()['label'] > threshold).to_list(), dtype=torch.long)
    groups = torch.tensor(df[['group', 'date']].drop_duplicates()['group'].to_list())
    sections = np.cumsum(df.groupby(['group', 'date']).count()['text']).tolist()[:-1]
    embeddings = list(torch.vsplit(embeddings, sections))
    for i in range(len(embeddings)):
        sample_size = min(embeddings[i].shape[0], max_samples) if max_samples else embeddings[i].shape[0]
        embeddings[i] = embeddings[i][torch.randperm(embeddings[i].shape[0])[:sample_size]]
    if balance_classes:
        sample_selector = torch.zeros_like(labels, dtype=bool)
        for g in groups.unique():
            selector = groups == g
            counts = labels[selector].unique(return_counts=True)[1]
            if len(counts) < 2:
                continue
            num_samples = counts.min()
            indices_0 = selector.nonzero().flatten()[labels[selector] == 0][sample(range(sum(labels[selector] == 0)), num_samples)]
            indices_1 = selector.nonzero().flatten()[labels[selector] == 1][sample(range(sum(labels[selector] == 1)), num_samples)]
            sample_selector[indices_0] = True
            sample_selector[indices_1] = True
        labels = labels[sample_selector]
        groups = groups[sample_selector]
        embeddings = [embeddings[i] for i in range(len(sample_selector)) if sample_selector[i]]
    if batched:
        embeddings = pad_sequence(embeddings, batch_first=True)
        if padding:
            embeddings = nn.functional.pad(embeddings, (0, 0, 0, max_samples - embeddings.shape[1]))
        return TensorDataset(embeddings, labels), groups
    return SimpleDataset(embeddings, labels), groups

def train_test(
        model: EmbeddingAggregator, 
        ds: Dataset, 
        groups: torch.Tensor, 
        batch_size: int = 1, 
        precision: str = 'bf16-mixed', 
        max_epochs: int = -1,
        max_time: Any | None = None, 
        deterministic: bool = False,
        predefined: bool | pd.DataFrame = False
):
    trainer = init_trainer(precision, early_stopping=True, logging={'name': 'aggregator', 'project': 'crisis-detector'}, max_epochs=max_epochs, max_time=max_time, deterministic=deterministic)
    if predefined == True:
        train_ds, test_ds, val_ds = predefined_split(ds, groups)
    elif predefined == False:
        train_ds, test_ds, val_ds = split_dataset(ds, groups, n_splits=10, validate=True)
    else:
        train_ds, test_ds, val_ds = predefined_split(ds, groups, predefined=predefined)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=10, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=10, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=10, pin_memory=True)
    model.train_dataloader_len = len(train_dl)
    model.max_epochs = max_epochs
    trainer.fit(model, train_dl, val_dl) 
    trainer.test(model, test_dl, 'best')
    if trainer.logger is not None:
        trainer.logger.experiment.finish()

def cross_validate(
        model: EmbeddingAggregator,
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
    init_state_dict = model.state_dict()
    for train_ds, test_ds, val_ds in tqdm(folds):
        model.load_state_dict(init_state_dict)
        if use_weights:
            class_ratio = train_ds[:][1].unique(return_counts=True)[1] / len(train_ds)
            model.init_loss_fn(class_ratio)

        trainer = train_model(model, train_ds, val_ds, precision, batch_size, max_epochs, max_time, num_workers, False, deterministic)

        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        test_results = trainer.test(dataloaders=test_dl, ckpt_path='best', verbose=False)[0]
        stats.append(test_results)
    stats = pd.DataFrame(stats)
    print(stats)
    print('Means:')
    print(stats.mean(axis=0))
    print('Standard deviation:')
    print(stats.std(axis=0))
    return stats

def main():
    deterministic = True
    end_to_end = False
    sample_size = 50
    batch_size = 512
    padding = False
    
    TEXTS_PATH = 'saved_objects/texts_no_sample_df.feather'
    EMBEDDINGS_PATH = 'saved_objects/embeddings_all.pt'
    # DATASET_PATH = 'saved_objects/day_embedding_ds_' + str(sample_size) + ('_batched' if batching else '') + ('_padded' if padding else '') + '.pt'

    if deterministic:
        seed_everything(42)

    if end_to_end or not os.path.isfile(TEXTS_PATH):
        dates = get_data_with_dates(get_verified_data())
        posts_df = load_data(dates['path'], dates['crisis_start'], drop_invalid=True)
        posts_df.to_feather(TEXTS_PATH)

        if deterministic:
            seed_everything(42)
    else:
        posts_df = pd.read_feather(TEXTS_PATH)
    
    if end_to_end or not os.path.isfile(EMBEDDINGS_PATH):
        embedder = TextEmbedder.load_from_checkpoint('saved_objects/pretrained_herbert.ckpt')
        tokenizer = AutoTokenizer.from_pretrained(embedder.pretrained_name)
        ds = SeriesDataset(posts_df['text'])
        collate_fn = lambda x: tokenizer(x, truncation=True, padding=True, max_length=256, return_tensors='pt')
        dl = DataLoader(ds, 128, num_workers=10, collate_fn=collate_fn, pin_memory=True)
        trainer = pl.Trainer(devices=1, precision='bf16-mixed', logger=False, deterministic=deterministic)
        embeddings = trainer.predict(embedder, dl)
        embeddings = torch.cat(embeddings, dim=0)

        with open(EMBEDDINGS_PATH, 'wb') as f:
            torch.save(embeddings, f)
        
        if deterministic:
            seed_everything(42, workers=True)
    else:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = torch.load(f)
    
    ds, groups = create_dataset(posts_df, embeddings, .02, sample_size, batch_size > 0, padding, balance_classes=True)
    model = MaskedAggregator(sample_size=sample_size)
    train_test(model, ds, groups, batch_size=batch_size, max_epochs=100, deterministic=deterministic)
    
    # cross_validate(MeanAggregator, {'sample_size': sample_size}, ds, groups, True, 5, batch_size=batch_size, num_workers=10, deterministic=deterministic)

if __name__ == '__main__':
    main()
