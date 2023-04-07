# %%
import torch
from torch import nn
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from tqdm import tqdm
from typing import List

torch.set_float32_matmul_precision('high')

# %%
DATA_DIR = 'crisis_data'
FILE_BLACKLIST = [
    'crisis_data/Afera Rywina.xlsx',
    'crisis_data/Ministerstwo Zdrowia_respiratory od handlarza bronią.xlsx',
    'crisis_data/Fake news_baza publikacji.xlsx'
]

files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f[-5:] == '.xlsx']
for f in FILE_BLACKLIST:
    files.remove(f)

# %%
def extract_data(filename: str):
    src_df = pd.read_excel(filename)

    if src_df['Kryzys'].hasnans:
        src_df['Kryzys'] = src_df['Kryzys'].notna()
    else:
        src_df['Kryzys'] = (src_df['Kryzys'] != 'NIE') & (src_df['Kryzys'] != 'Nie')
    if src_df['Kryzys'].nunique() != 2:
        raise RuntimeError(f'Crisis column data error in file {filename}.')
    
    new_cols = ['brak', 'negatywny', 'neutralny', 'pozytywny']
    new_cols_ex = [c for c in new_cols if c in src_df['Wydźwięk'].unique().tolist()]
    src_df[new_cols_ex] = pd.get_dummies(src_df['Wydźwięk'])
    for col in new_cols:
        if col not in src_df.columns:
            src_df[col] = 0

    df = src_df[['Data wydania', 'Kryzys']].groupby(['Data wydania']).any()
    df = df.join(src_df[['Data wydania'] + new_cols].groupby(['Data wydania']).sum())

    df = df.reindex(pd.date_range(df.index.min(), df.index.max()))
    df[new_cols] = df[new_cols].fillna(0)
    df['Kryzys'] = df['Kryzys'].fillna(method='ffill') & df['Kryzys'].fillna(method='bfill')

    df['suma'] = df[new_cols].sum(axis=1)

    X = torch.tensor(df.drop('Kryzys', axis=1).values, dtype=torch.float32)
    zeros = X == 0
    X[1:] = X[1:] / X[:-1]
    X[0] = 1.
    X = X.pow(2)
    X = (X - 1) / (X + 1)
    X[zeros & X.isnan()] = 0.
    X[X.isnan()] = 1.

    y = torch.tensor(df['Kryzys'].values, dtype=torch.long)
    
    return X, y
    

# %%
def extract_raw_data(filename: str):
    src_df = pd.read_excel(filename)

    if src_df['Kryzys'].hasnans:
        src_df['Kryzys'] = src_df['Kryzys'].notna()
    else:
        src_df['Kryzys'] = (src_df['Kryzys'] != 'NIE') & (src_df['Kryzys'] != 'Nie')
    if src_df['Kryzys'].nunique() != 2:
        raise RuntimeError(f'Crisis column data error in file {filename}.')
    
    new_cols = ['brak', 'negatywny', 'neutralny', 'pozytywny']
    new_cols_ex = [c for c in new_cols if c in src_df['Wydźwięk'].unique().tolist()]
    src_df[new_cols_ex] = pd.get_dummies(src_df['Wydźwięk'])
    for col in new_cols:
        if col not in src_df.columns:
            src_df[col] = 0

    df = src_df[['Data wydania', 'Kryzys']].groupby(['Data wydania']).any()
    df = df.join(src_df[['Data wydania'] + new_cols].groupby(['Data wydania']).sum())

    df = df.reindex(pd.date_range(df.index.min(), df.index.max()))
    df[new_cols] = df[new_cols].fillna(0)
    df['Kryzys'] = df['Kryzys'].fillna(method='ffill') & df['Kryzys'].fillna(method='bfill')

    df['suma'] = df[new_cols].sum(axis=1)

    X = torch.tensor(df.drop('Kryzys', axis=1).values, dtype=torch.long)
    y = torch.tensor(df['Kryzys'].values, dtype=torch.long)
    
    return X, y
    

# %%
def transform_data(tensors: List):
    t2 = []
    for X, y in tensors:
        newX, newy = torch.zeros_like(X, dtype=torch.float32), y.clone().detach()
        zeros = X == 0
        newX[1:] = X[1:] / X[:-1]
        newX[0] = 1.
        newX = newX.pow(2)
        newX = (newX - 1) / (newX + 1)
        newX[zeros & newX.isnan()] = 0.
        newX[newX.isnan()] = 1.
        t2.append((newX, newy))
    return t2

# %%
def transform_data_simple(tensors: List):
    t2 = []
    for X, y in tensors:
        newX = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
        newy = y.clone().detach()
        t2.append((newX, newy))
    return t2

# %%
def create_datasets(tensors, sequence_len: int = 30):
    newX, newy = [], []
    for (X, y) in tensors:
        for i in range(len(X) - sequence_len + 1):
            newX.append(X[i:i+sequence_len])
            newy.append(y[i+sequence_len-1])
    newX, newy = torch.stack(newX), torch.tensor(newy, dtype=torch.long)
    ds = TensorDataset(newX, newy)
    return random_split(ds, (.7, .15, .15))
    

# %%
class MyModel(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, class_weight: torch.Tensor) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.loss_fn = nn.CrossEntropyLoss(class_weight)
        self.f1 = torchmetrics.F1Score('binary', average='macro')
        self.acc = torchmetrics.Accuracy('binary')

        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, dropout=0.2)
            ),
            nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, self.n_classes),
                nn.Softmax(dim=-1)
            )
        ])
    
    def forward(self, x):
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(optimizer, 'min')
        return optimizer

# %%
class RandomModel(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, class_weight: torch.Tensor) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.loss_fn = nn.CrossEntropyLoss(class_weight)
        self.f1 = torchmetrics.F1Score('binary', average='macro')
        self.acc = torchmetrics.Accuracy('binary')

        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
            ),
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.n_classes),
                nn.Softmax(dim=-1)
            )
        ])
    
    def forward(self, x):
        x, _ = self.nets[0](x)
        x = self.nets[1](x[:,-1,:])
        return torch.nn.functional.softmax(torch.rand_like(x, requires_grad=True), dim=-1)

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
        precision = torchmetrics.functional.precision(torch.argmax(y_pred, -1), y)
        recall = torchmetrics.functional.recall(torch.argmax(y_pred, -1), y)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(optimizer, 'min')
        return optimizer

# %%
# tensors = [extract_raw_data(f) for f in tqdm(files)]

# with open('other_data/tensors.pt', 'wb') as f:
#     torch.save(tensors, f)

with open('other_data/tensors.pt', 'rb') as f:
    tensors = torch.load(f)

# %%
tensors = transform_data_simple(tensors)

# %%
train_ds, val_ds, test_ds = create_datasets(tensors)

weight = torch.unique(train_ds[:][1], sorted=True, return_counts=True)[1] / len(train_ds)
# weight = torch.flip(weight, dims=[0])
weight = torch.nn.functional.softmax(1 / weight)

# %%
BATCH_SIZE = 256
CHECKPOINT_PATH = 'checkpoints/checkpoint.ckpt'

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True)

model = MyModel(5, 30, 2, weight)
if os.path.isfile(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

checkpoint_callback = ModelCheckpoint(
    # dirpath='checkpoints/',
    # filename="checkpoint",
    monitor='val_f1',
    mode='max',
    save_top_k=1
)
trainer = pl.Trainer(
    accelerator='gpu',
    callbacks=[
        EarlyStopping(monitor="val_f1", mode="max", patience=20),
        checkpoint_callback
    ]
)
trainer.fit(model, train_dl, val_dl)

# %%
trainer.test(model, test_dl, ckpt_path="best")
