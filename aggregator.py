import math
from typing import Any, Optional, Tuple, Iterable
import os
import numpy as np
import pandas as pd
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
from transformers import AutoTokenizer

from data_tools import get_data_with_dates, get_verified_data, SeriesDataset, SimpleDataset
from training_tools import split_dataset, fold_dataset, train_model
from embedder import TextEmbedder

class EmbeddingAggregator(pl.LightningModule):
    def __init__(
            self,
            sample_size: int = 0,
            embedding_dim: int = 768,
            lr: float = 1e-3,
            weight_decay: float = 0.01,
            class_ratios: torch.Tensor | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.sample_size = sample_size
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = .5 / class_ratios if class_ratios is not None else None

        self.loss_fn = nn.CrossEntropyLoss(self.class_weights)
        self.f1 = torchmetrics.F1Score('multiclass', num_classes=2, average=None)
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=2, average='micro')
        self.prec = torchmetrics.Precision('multiclass', num_classes=2, average=None)
        self.rec = torchmetrics.Recall('multiclass', num_classes=2, average=None)
        
        self.classifier = nn.Sequential(
            nn.Dropout(.1),
            nn.Linear(self.embedding_dim, 384),
            nn.Dropout(.1),
            nn.Tanh(),
            nn.Linear(384, 2),
            nn.Softmax(dim=-1)
        )
    
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
        self.log('val_loss', self.loss_fn(y_pred, y).item(), on_epoch=True)
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        metrics = {
            'val_acc': self.acc.compute().item(),
            'val_f1': self.f1.compute().mean().item()
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
        self.log('test_loss', self.loss_fn(y_pred, y).item(), on_epoch=True)
    
    @torch.no_grad()
    def on_test_epoch_end(self) -> None:
        metrics = {
            'test_acc': self.acc.compute().item()
        }
        f1 = self.f1.compute()
        metrics['test_f1_neg'], metrics['test_f1_pos'] = f1[0].item(), f1[1].item()
        prec = self.prec.compute()
        metrics['test_precision_neg'], metrics['test_precision_pos'] = prec[0].item(), prec[1].item()
        rec = self.rec.compute()
        metrics['test_recall_neg'], metrics['test_recall_pos'] = rec[0].item(), rec[1].item()
        self.log_dict(metrics)
        self.acc.reset()
        self.f1.reset()
        self.prec.reset()
        self.rec.reset()
    
    def configure_optimizers(self) -> Any:
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

class MeanAggregator(EmbeddingAggregator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        print(x.numel())
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
    def __init__(self, n_heads: int = 8, transformer_layers: int = 6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads
        self.transformer_layers = transformer_layers

        self.positional_encoding = PositionalEncoding(self.embedding_dim, max_len=self.sample_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.embedding_dim, self.n_heads, activation=nn.functional.leaky_relu, batch_first=True),
            self.transformer_layers
        )
    
    def forward(self, x: torch.Tensor):
        x = self.positional_encoding(x)
        x = self.transformer(x)[:, -1]
        return x

def extract_data(
        filename: str,
        crisis_start: pd.Timestamp,
        window_size: int | Tuple[int, int] = 30,
        drop_invalid: bool = False
) -> pd.DataFrame | None:
    src_df = pd.read_excel(filename)

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

def create_dataset(df: pd.DataFrame, embeddings: torch.Tensor, threshold: float = 0., max_samples: int = 0, batched: bool = True, padding: bool = False):
    assert (batched and max_samples) or not padding
    labels = torch.tensor((df[['group', 'date', 'label']].groupby(['group', 'date']).mean()['label'] > threshold).to_list(), dtype=torch.long)
    groups = torch.tensor(df[['group', 'date']].drop_duplicates()['group'].to_list())
    sections = np.cumsum(df.groupby(['group', 'date']).count()['text']).tolist()[:-1]
    embeddings = list(torch.vsplit(embeddings, sections))
    for i in range(len(embeddings)):
        sample_size = min(embeddings[i].shape[0], max_samples) if max_samples else embeddings[i].shape[0]
        embeddings[i] = embeddings[i][torch.randperm(embeddings[i].shape[0])[:sample_size]]
    if batched:
        embeddings = pad_sequence(embeddings, batch_first=True)
        if padding:
            embeddings = nn.functional.pad(embeddings, (0, 0, 0, max_samples - embeddings.shape[1]))
        return TensorDataset(embeddings, labels), groups
    return SimpleDataset(embeddings, labels), groups

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
            model_params['class_ratios'] = class_ratio
        model = model_class(**model_params)

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
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        ds = SeriesDataset(posts_df['text'])
        collate_fn = lambda x: tokenizer(x, truncation=True, padding=True, max_length=256, return_tensors='pt')
        dl = DataLoader(ds, 64, num_workers=10, collate_fn=collate_fn, pin_memory=True)
        model = TextEmbedder.load_from_checkpoint('saved_objects/finetuned-xlm-roberta-base.ckpt')
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
    
    ds, groups = create_dataset(posts_df, embeddings, 0., sample_size, batch_size > 0, padding)
    # print(sum(ds[:][1]) / len(ds))
    
    cross_validate(TransformerAggregator, {'sample_size': sample_size}, ds, groups, True, 5, batch_size=batch_size, num_workers=10, deterministic=deterministic)

if __name__ == '__main__':
    main()
