from typing import Any, Optional, Tuple, Iterable
import os
import numpy as np
import pandas as pd
from warnings import warn
import tqdm

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torchmetrics
from transformers import AutoTokenizer

from data_tools import get_data_with_dates, get_verified_data, SeriesDataset
from embedder import TextEmbedder

class AggregatorBackbone(nn.Module):
    def __init__(self, sample_size: int = 0, embedding_dim: int = 768, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
        self.embedding_dim = embedding_dim

class EmbeddingAggregator(pl.LightningModule):
    def __init__(
            self,
            backbone: AggregatorBackbone,
            lr: float = 1e-3,
            weight_decay: float = 0.01,
            class_ratios: torch.Tensor | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = .5 / class_ratios if class_ratios else None

        self.loss_fn = nn.CrossEntropyLoss(self.class_weights)
        self.f1 = torchmetrics.F1Score('multiclass', num_classes=2, average=None)
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=2, average='micro')
        self.prec = torchmetrics.Precision('multiclass', num_classes=2, average=None)
        self.rec = torchmetrics.Recall('multiclass', num_classes=2, average=None)
        
        self.model = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(.1),
            nn.Linear(backbone.embedding_dim, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x) -> Any:
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, y = batch
        y_pred = self.classifier(self(X))
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        X, y = batch
        y_pred = self.classifier(self(X))
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
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        X, y = batch
        y_pred = self.classifier(self(X))
        self.acc.update(torch.argmax(y_pred, -1), y)
        self.f1.update(torch.argmax(y_pred, -1), y)
        self.prec.update(torch.argmax(y_pred, -1), y)
        self.rec.update(torch.argmax(y_pred, -1), y)
        self.log('test_loss', self.loss_fn(y_pred, y), on_epoch=True)
    
    def on_test_epoch_end(self) -> None:
        metrics = {
            'test_acc': self.acc.compute()
        }
        metrics['test_f1_neg'], metrics['test_f1_pos'] = self.f1.compute()
        metrics['test_precision_neg'], metrics['test_precision_pos'] = self.prec.compute()
        metrics['test_recall_neg'], metrics['test_recall_pos'] = self.rec.compute()
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

class MeanBackbone(AggregatorBackbone):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return torch.mean(x, dim=1)

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
        df = extract_data(fname, date, drop_invalid=drop_invalid)
        if df is None:
            continue
        df['group'] = i
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def create_dataset(df: pd.DataFrame, embeddings: torch.Tensor, sample_size: int = 0, batched: bool = True):
    sections = np.cumsum(df.groupby(['group', 'date']).count()['text']).tolist()[:-1]
    labels = df[['group', 'date', 'label']].groupby(['group', 'date']).mean()
    embeddings = torch.vsplit(embeddings, sections)
    if batched:
        assert sample_size == 0
        embeddings = pad_sequence(embeddings, batch_first=True)


def main():
    TEXTS_PATH = 'saved_objects/texts_no_sample_df.feather'
    EMBEDDINGS_PATH = 'saved_objects/token_ds.pt'

    deterministic = True
    end_to_end = False
    sample_size = 50

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
    

if __name__ == '__main__':
    main()
