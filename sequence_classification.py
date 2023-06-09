# %%
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from typing import List, Iterable, Tuple
from sklearn.model_selection import GroupKFold
from warnings import warn
import json

torch.set_float32_matmul_precision('high')
torch.random.manual_seed(42)

# %%
DATA_DIR = 'crisis_data'
# FILE_BLACKLIST = [
#     'crisis_data/Afera Rywina.xlsx',
#     'crisis_data/Ministerstwo Zdrowia_respiratory od handlarza bronią.xlsx',
#     'crisis_data/Fake news_baza publikacji.xlsx'
# ]
FILE_BLACKLIST = [
    'crisis_data/Jan Szyszko_Córka leśniczego.xlsx',
    'crisis_data/Komenda Główna Policji.xlsx',
    'crisis_data/Ministerstwo Zdrowia_respiratory od handlarza bronią.xlsx',
    'crisis_data/Polska Grupa Energetyczna.xlsx',
    'crisis_data/Polski Związek Kolarski.xlsx',
    'crisis_data/Zbój_energetyk.xlsx'
]

files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f[-5:] == '.xlsx']
for f in FILE_BLACKLIST:
    files.remove(f)

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
def clip_date_range(index: pd.DatetimeIndex, crisis_start: pd.Timestamp):
    return pd.date_range(max(index.min(), crisis_start - pd.Timedelta(days=60)), min(index.max(), crisis_start + pd.Timedelta(days=29)))

def extract_data(filename: str, crisis_start: pd.Timestamp):
    src_df = pd.read_excel(filename)
    
    new_cols = ['brak', 'negatywny', 'neutralny', 'pozytywny']
    new_cols_ex = [c for c in new_cols if c in src_df['Wydźwięk'].unique().tolist()]
    src_df[new_cols_ex] = pd.get_dummies(src_df['Wydźwięk'])
    for col in new_cols:
        if col not in src_df.columns:
            src_df[col] = 0

    df = src_df[['Data wydania'] + new_cols].groupby(['Data wydania']).sum()

    df = df.reindex(clip_date_range(df.index, crisis_start))
    df[new_cols] = df[new_cols].fillna(0)

    df['suma'] = df[new_cols].sum(axis=1)
    labels = df.index >= crisis_start
    if np.unique(labels).shape[0] != 2:
        warn(f'Samples from only 1 class in {filename}.')
    if df.shape[0] == 0:
        warn(f'No data after clipping for {filename}.')

    X = torch.tensor(df.values, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    
    return X, y

# %%
def transform_data(tensors: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Xs, ys, ids = [], [], []
    for idx, (X, y) in enumerate(tensors):
        newX = torch.zeros((X.shape[0], 14), dtype=torch.float32)
        newy = y.clone().detach()
        newX[:, :5] = torch.tensor(StandardScaler().fit_transform(X))
        zeros = X == 0
        newX[1:, 5:10] = X[1:] / X[:-1]
        newX[0, 5:10] = 1.
        newX[:, 5:10] = newX[:, 5:10].pow(2)
        newX[:, 5:10] = (newX[:, 5:10] - 1) / (newX[:, 5:10] + 1)
        newX[:, 5:10][zeros & newX[:, 5:10].isnan()] = 0.
        newX[:, 5:10][newX[:, 5:10].isnan()] = 1.
        newX[:, 10:] = X[:, :4] / X[:, 4:]
        newX[:, 10:][newX[:, 10:].isnan()] = 0.
        Xs.append(newX)
        ys.append(newy)
        ids.append(torch.full_like(newy, idx))
    return torch.cat(Xs), torch.cat(ys), torch.cat(ids)

# %%
def create_dataset(X: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, sequence_len: int = 30):
    newX, newy, newgroups = [], [], []
    for idx in groups.unique():
        Xi = X[groups == idx]
        yi = y[groups == idx]
        for i in range(len(yi) - sequence_len + 1):
            newX.append(Xi[i:i+sequence_len])
            newy.append(yi[i+sequence_len-1])
            newgroups.append(idx)
    newX, newy, newgroups = torch.stack(newX), torch.tensor(newy, dtype=torch.long), torch.tensor(newgroups)
    ds = TensorDataset(newX, newy)
    return ds, newgroups
    
# %%
def split_dataset(dataset: Dataset, groups: torch.Tensor, n_splits: int = 10, return_groups: bool = False
                  ) -> Tuple[Dataset, Dataset, Dataset] | Tuple[Tuple[Dataset, Dataset, Dataset], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    fold = GroupKFold(n_splits)
    folds = list(fold.split(dataset, groups=groups))
    val_idx = folds[1][1]
    train_idx = list(set(folds[0][0]) - set(val_idx))
    test_idx = folds[0][1]
    if return_groups:
        return (Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)), (groups[train_idx], groups[val_idx], groups[test_idx])
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

# %%
def fold_dataset(dataset: Dataset, groups: torch.Tensor, n_splits: int = 5) -> List[Tuple[Dataset, Dataset, Dataset]]:
    fold = GroupKFold(n_splits)
    splits = list(fold.split(dataset, groups=groups))
    test_splits = [split[1] for split in splits]
    val_splits = test_splits[1:] + test_splits[:1]
    splits = [(list(set(train_idx) - set(val_idx)), val_idx, test_idx) for ((train_idx, test_idx), val_idx) in zip(splits, val_splits)]
    return [(Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)) for train_idx, test_idx, val_idx in splits]

# %%
class MyModel(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.f1 = torchmetrics.F1Score('binary', average='macro')
        self.acc = torchmetrics.Accuracy('binary')

        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
            ),
            nn.Sequential(
                nn.Dropout(0.1),
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
    def __init__(self, n_classes: int, class_probs: torch.Tensor) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.class_probs = class_probs
        self.f1 = torchmetrics.F1Score('binary', average='macro')
        self.acc = torchmetrics.Accuracy('binary')
    
    def forward(self, x):
        preds = torch.randint(0, self.n_classes, (x.shape[0],), device=self.device)
        return torch.nn.functional.one_hot(preds).float()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        acc = self.acc(torch.argmax(y_pred, -1), y)
        score = self.f1(torch.argmax(y_pred, -1), y)
        precision = torchmetrics.functional.precision(torch.argmax(y_pred, -1), y, 'binary', average='macro')
        recall = torchmetrics.functional.recall(torch.argmax(y_pred, -1), y, 'binary', average='macro')
        self.log('test_acc', acc, on_epoch=True)
        self.log('test_f1', score, on_epoch=True)
        self.log('test_precision', precision, on_epoch=True)
        self.log('test_recall', recall, on_epoch=True)

# %%
class ConstantModel(pl.LightningModule):
    def __init__(self, n_classes: int, return_value: int = 0) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.f1 = torchmetrics.F1Score('binary', average='macro')
        self.acc = torchmetrics.Accuracy('binary')

        self.output_tensor = torch.zeros(n_classes, dtype=torch.float32)
        self.output_tensor[return_value] = 1.
        self.output_tensor.requires_grad = True
    
    def forward(self, x):
        return torch.tile(self.output_tensor, (x.shape[0], 1)).to(self.device)
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        acc = self.acc(torch.argmax(y_pred, -1), y)
        score = self.f1(torch.argmax(y_pred, -1), y)
        precision = torchmetrics.functional.precision(torch.argmax(y_pred, -1), y, 'binary', average='macro')
        recall = torchmetrics.functional.recall(torch.argmax(y_pred, -1), y, 'binary', average='macro')
        self.log('test_acc', acc, on_epoch=True)
        self.log('test_f1', score, on_epoch=True)
        self.log('test_precision', precision, on_epoch=True)
        self.log('test_recall', recall, on_epoch=True)

# %%
def train_model(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset) -> pl.LightningModule:
    BATCH_SIZE = 256

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True)

    model = MyModel(14, 80, 2)

    checkpoint_callback = ModelCheckpoint(
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
    trainer.test(model, test_dl, ckpt_path="best")
    return trainer.lightning_module

# %%
def cross_validate(folds: Iterable[Tuple[Dataset, Dataset, Dataset]], batch_size: int = 512) -> pd.DataFrame:
    stats = []
    for train_ds, val_ds, test_ds in folds:
        weight = torch.unique(train_ds[:][1], sorted=True, return_counts=True)[1] / len(train_ds)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)

        model = MyModel(14, 80, 2)

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
        stats += trainer.test(model, test_dl, ckpt_path="best")
        # stats += trainer.test(model, test_dl)
    return pd.DataFrame(stats)

# %%
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

# %%
@torch.no_grad()
def test_shift(model: pl.LightningModule, test_ds: Dataset) -> Tuple[float, float]:
    X, y = test_ds[:]
    y_pred = torch.argmax(model(X), dim=-1)
    print(y)
    print(y_pred)
    shifts = get_shifts(y, y_pred).abs().float()
    return shifts.mean().item(), shifts.std().item()

# %%
def save_shift(model: pl.LightningModule, test_ds: Dataset, crisis_names: Iterable[str], filename: str) -> None:
    X, y = test_ds[:]
    y_pred = torch.argmax(model(X), dim=-1)
    shifts = get_shifts(y, y_pred)
    d = {}
    for i, name in enumerate(crisis_names):
        d[name] = shifts[i].item()
    with open(filename, 'w') as f:
        json.dump(d, f)

# %%
seed_everything(42)

# %%
crisis = pd.read_excel('crisis_data/Daty_kryzysów.xlsx').dropna()
crisis = crisis[~crisis['Plik'].apply(lambda x: os.path.join(DATA_DIR, x) in FILE_BLACKLIST)]
# tensors = [extract_data(os.path.join(DATA_DIR, row.Plik), row.Data) for _, row in tqdm(crisis.iterrows(), total=crisis.shape[0])]
# tensors = [(X, y) for X, y in tensors if y.shape[0] > 0]

# with open('other_data/tensors.pt', 'wb') as f:
#     torch.save(tensors, f)

with open('other_data/tensors.pt', 'rb') as f:
    tensors = torch.load(f)

# %%
X, y, groups = transform_data(tensors)
ds, groups = create_dataset(X, y, groups)
# (train_ds, val_ds, test_ds), (_, _, file_ids)  = split_dataset(ds, groups, return_groups=True)
# file_ids = file_ids.unique(sorted=False).flip(dims=(0,))
# model = train_model(train_ds, val_ds, test_ds)
# test_shift(model, test_ds)
# save_shift(model, test_ds, list(crisis['Plik'].iloc[file_ids]), 'other_data/shifts.json')
folds = fold_dataset(ds, groups)
# for (train_ds, val_ds, test_ds) in folds:
#     print(sum(train_ds[:][1]).item() / len(train_ds), sum(val_ds[:][1]).item() / len(val_ds), sum(test_ds[:][1]).item() / len(test_ds))

# %%
df = cross_validate(folds)
print(df)
print('Means:')
print(df.mean(axis=0))
print('Standard deviation:')
print(df.std(axis=0))
exit(0)

# %%

weight = torch.unique(ds[:][1], sorted=True, return_counts=True)[1] / len(ds)
# weight = torch.flip(weight, dims=[0])
weight = torch.pow(1 - weight, .8)
weight /= sum(weight)

# %%
BATCH_SIZE = 256
CHECKPOINT_PATH = 'checkpoints/checkpoint.ckpt'

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True)

model = MyModel(14, 80, 2, weight)
# model = RandomModel(2, torch.unique(train_ds[:][1], sorted=True, return_counts=True)[1] / len(train_ds))
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
# trainer.test(model, test_dl)
