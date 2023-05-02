from typing import List, Tuple, Iterable, Any
from warnings import warn
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import lightning as pl
from lightning.pytorch import seed_everything
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler

from embedder import TextEmbedder
from data_tools import load_data, SeriesDataset, get_all_data, get_data_with_dates
from training_tools import split_dataset, cross_validate, train_model, test_model

torch.set_float32_matmul_precision('high')

def add_embeddings(days_df: pd.DataFrame, text_df: pd.DataFrame, embeddings: List[torch.Tensor] | torch.Tensor) -> pd.DataFrame:
    if type(embeddings) == list:
        embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.numpy()
    sections = np.cumsum(text_df.groupby(['group', 'Data wydania']).count()['text']).tolist()[:-1]
    day_embeddings = np.stack([np.mean(t, axis=0) for t in np.vsplit(embeddings, sections)], axis=0)
    embedding_df = text_df[['group', 'Data wydania']].drop_duplicates().reset_index(drop=True)
    embedding_df['embedding'] = day_embeddings.tolist()
    days_df = days_df.join(embedding_df.set_index(['group', 'Data wydania']), ['group', 'Data wydania'], how='left')
    days_df.loc[:, 'embedding'].iloc[days_df['embedding'].isna()] = pd.Series(np.zeros((days_df['embedding'].isna().sum(), 768)).tolist())
    return days_df

def create_dataset(df: pd.DataFrame, sequence_len: int = 30) -> Tuple[Dataset, torch.Tensor]:
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

class MyModel(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, limit_inputs: int = 0) -> None:
        super().__init__()
        self.limit_inputs = bool(limit_inputs)
        self.input_dim = limit_inputs if self.limit_inputs else input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.f1 = torchmetrics.F1Score('multiclass', num_classes=2, average='macro')
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=2, average='macro')
        self.prec = torchmetrics.Precision('multiclass', num_classes=2, average='macro')
        self.rec = torchmetrics.Recall('multiclass', num_classes=2, average='macro')

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
                nn.Linear(self.hidden_dim, self.n_classes),
                nn.Softmax(dim=-1)
            )
        ])
    
    def forward(self, x):
        if self.limit_inputs:
            x = x[..., :self.input_dim]
        x, _ = self.nets[0](x)
        x = self.nets[1](x[:,-1,:])
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        acc = self.acc(torch.argmax(y_pred, -1), y)
        f1 = self.f1(torch.argmax(y_pred, -1), y)
        loss = self.loss_fn(y_pred, y)
        self.validation_step_losses.append(loss)
        self.log('val_acc', acc, on_epoch=True)
        self.log('val_f1', f1, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True)
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        acc = self.acc(torch.argmax(y_pred, -1), y)
        score = self.f1(torch.argmax(y_pred, -1), y)
        loss = self.loss_fn(y_pred, y)
        precision = self.prec(torch.argmax(y_pred, -1), y)
        recall = self.rec(torch.argmax(y_pred, -1), y)
        self.log('test_acc', acc, on_epoch=True)
        self.log('test_f1', score, on_epoch=True)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_precision', precision, on_epoch=True)
        self.log('test_recall', recall, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        self.validation_step_losses = []

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_losses).mean(dim=0)
        self.scheduler.step(loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(optimizer, 'min')
        return optimizer

def get_shifts(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    sequence_idx = (torch.diff(y_true, prepend=torch.tensor([1]), append=torch.tensor([0])) == -1).nonzero().squeeze()
    crisis_idx = (torch.diff(y_true, prepend=torch.tensor([0])) == 1).nonzero().squeeze()
    pred_start = torch.diff(y_pred, prepend=torch.tensor([0])) == 1
    shifts = torch.zeros_like(crisis_idx)
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
def test_shift(model: pl.LightningModule, test_ds: Dataset) -> Tuple[float, float]:
    X, y = test_ds[:]
    y_pred = torch.argmax(model(X), dim=-1)
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
    end_to_end = True

    if deterministic:
        seed_everything(42, workers=True)

    DAYS_DF_PATH = 'saved_objects/days_df.feather'
    POSTS_DF_PATH = 'saved_objects/posts_df.feather'
    EMBEDDINGS_PATH = 'saved_objects/embeddings.pt'

    if end_to_end or not (os.path.isfile(DAYS_DF_PATH) and os.path.isfile(POSTS_DF_PATH)):

        data = get_data_with_dates(get_all_data())

        days_df, text_df = load_data(data['path'].to_list(), data['Data'].to_list(), 200, True)
        days_df.to_feather(DAYS_DF_PATH)
        text_df.to_feather(POSTS_DF_PATH)
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
    else:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = torch.load(f)

    days_df = add_embeddings(days_df, text_df, embeddings)
    ds, groups = create_dataset(days_df)

    train_ds, test_ds, val_ds = split_dataset(ds, groups)
    model = MyModel(782, 128, 2)
    trainer = train_model(model, train_ds, val_ds, precision='32', deterministic=deterministic)
    test_model(test_ds, trainer=trainer, precision='32', deterministic=deterministic)

    print(test_shift(model, test_ds))

    # cross_validate(MyModel, (782, 128, 2), ds, groups, n_splits=5, precision='32', deterministic=deterministic)

if __name__ == '__main__':
    main()
