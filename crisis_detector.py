from typing import List, Tuple, Iterable, Any
from warnings import warn
import os
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
from data_tools import load_data, SeriesDataset
from training_tools import split_dataset, cross_validate

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
        self.f1 = torchmetrics.F1Score('binary', average='macro')
        self.acc = torchmetrics.Accuracy('binary')

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
        precision = torchmetrics.functional.precision(torch.argmax(y_pred, -1), y, 'binary', average='macro')
        recall = torchmetrics.functional.recall(torch.argmax(y_pred, -1), y, 'binary', average='macro')
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

def main():
    deterministic = True
    end_to_end = True

    if deterministic:
        seed_everything(42, workers=True)

    DAYS_DF_PATH = 'data/saved_objects/days_df.feather'
    POSTS_DF_PATH = 'data/saved_objects/posts_df.feather'
    EMBEDDINGS_PATH = 'data/saved_objects/embeddings.pt'

    if end_to_end or not (os.path.isfile(DAYS_DF_PATH) and os.path.isfile(POSTS_DF_PATH)):
        CRISIS_DIR = 'data/crisis_data'
        CRISIS_FILES_BLACKLIST = [
            'Crisis Detector - lista wątków_.docx',
            'Daty_kryzysów.xlsx',
            'Jan Szyszko_Córka leśniczego.xlsx',
            'Komenda Główna Policji.xlsx',
            'Ministerstwo Zdrowia_respiratory od handlarza bronią.xlsx',
            'Polska Grupa Energetyczna.xlsx',
            'Polski Związek Kolarski.xlsx',
            'Zbój_energetyk.xlsx'
        ]

        crisis = pd.read_excel(os.path.join(CRISIS_DIR, 'Daty_kryzysów.xlsx')).dropna()
        crisis = crisis[~crisis['Plik'].apply(lambda x: x in CRISIS_FILES_BLACKLIST)]

        days_df, text_df = load_data(crisis['Plik'].apply(lambda x: os.path.join(CRISIS_DIR, x)).to_list(), crisis['Data'].to_list(), 100)
        days_df.to_feather(DAYS_DF_PATH)
        text_df.to_feather(POSTS_DF_PATH)
    else:
        days_df = pd.read_feather(DAYS_DF_PATH)
        text_df = pd.read_feather(POSTS_DF_PATH)

    if end_to_end or not os.path.isfile(EMBEDDINGS_PATH):
        tokenizer = AutoTokenizer.from_pretrained('sdadas/polish-distilroberta')
        ds = SeriesDataset(text_df['text'])
        collate_fn = lambda x: tokenizer(x, truncation=True, padding=True, max_length=256, return_tensors='pt')
        dl = DataLoader(ds, 256, num_workers=10, collate_fn=collate_fn, pin_memory=True)
        model = TextEmbedder.load_from_checkpoint('checkpoints/epoch=0-step=7313.ckpt')
        # model = TextEmbedder('sdadas/polish-distilroberta')
        trainer = pl.Trainer(accelerator='gpu', precision='bf16-mixed', logger=None, deterministic=deterministic)
        embeddings = trainer.predict(model, dl)
        embeddings = torch.cat(embeddings, dim=0)

        with open(EMBEDDINGS_PATH, 'wb') as f:
            torch.save(embeddings, f)
    else:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = torch.load(f)

    days_df = add_embeddings(days_df, text_df, embeddings)
    ds, groups = create_dataset(days_df)

    # train_ds, test_ds, val_ds = split_dataset(ds, groups)
    # model = MyModel(782, 128, 2)
    # trainer = train_model(model, train_ds, val_ds, deterministic=deterministic)
    # results = test_model(trainer, test_ds)

    cross_validate(MyModel, (782, 128, 2), ds, groups, n_splits=5, precision='32', deterministic=deterministic)

if __name__ == '__main__':
    main()