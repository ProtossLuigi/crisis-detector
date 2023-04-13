# %%
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
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
from typing import List, Iterable, Tuple
from sklearn.model_selection import GroupKFold

torch.set_float32_matmul_precision('high')
torch.random.manual_seed(42)

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
def split_dataset(dataset: Dataset, groups: torch.Tensor, lengths: Iterable[int], shuffle: bool = True) -> List[Dataset]:
    group_idx, group_lengths = groups.unique(return_counts=True)
    lengths = torch.tensor(lengths) / sum(lengths) * len(dataset)
    if shuffle:
        new_idx = torch.randperm(len(group_lengths))
        group_idx = group_idx[new_idx]
    splits = torch.zeros((len(lengths), len(dataset)), dtype=bool)
    li = 0
    for g in group_idx:
        if sum(splits[li]) + group_lengths[g] / 2 < lengths[li] or li == len(lengths) - 1:
            splits[li] |= groups == g
        else:
            li += 1
            splits[li] |= groups == g
    splits = torch.stack(torch.nonzero(split).squeeze() for split in splits)
    subsets = [Subset(dataset, split) for split in splits]
    return subsets

# %%
def fold_dataset(dataset: Dataset, groups: torch.Tensor, n_splits: int = 5) -> List[Tuple[Dataset, Dataset, Dataset]]:
    fold = GroupKFold(n_splits)
    splits = list(fold.split(dataset, groups=groups))
    test_splits = [split[1] for split in splits]
    val_splits = [test_splits[-1]] + test_splits[:-1]
    splits = [(list(set(train_idx) - set(val_idx)), val_idx, test_idx) for ((train_idx, test_idx), val_idx) in zip(splits, val_splits)]
    return [(Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)) for train_idx, test_idx, val_idx in splits]

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
def cross_validate(folds: Iterable[Tuple[Dataset, Dataset, Dataset]], batch_size: int = 512):
    stats = []
    for train_ds, val_ds, test_ds in folds:
        weight = torch.unique(train_ds[:][1], sorted=True, return_counts=True)[1] / len(train_ds)
        weight = torch.pow(1 - weight, 1)
        weight /= sum(weight)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)

        model = MyModel(14, 80, 2, weight)

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
# tensors = [extract_raw_data(f) for f in tqdm(files)]

# with open('other_data/tensors.pt', 'wb') as f:
#     torch.save(tensors, f)

with open('other_data/tensors.pt', 'rb') as f:
    tensors = torch.load(f)

# %%
X, y, groups = transform_data(tensors)
ds, groups = create_dataset(X, y, groups)
print(len(ds))
# train_ds, val_ds, test_ds = split_dataset(ds, groups, [.7, .15, .15])
folds = fold_dataset(ds, groups)
for (train_ds, val_ds, test_ds) in folds:
    print(len(train_ds), len(val_ds), len(test_ds))

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
