from typing import List, Tuple, Iterable, Any
from warnings import warn
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from transformers import AutoTokenizer, RobertaModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

torch.set_float32_matmul_precision('high')

def clip_date_range(index: pd.DatetimeIndex, crisis_start: pd.Timestamp | None = None, window_size: int | Tuple[int, int] | None = None) -> pd.DatetimeIndex:
    if type(window_size) == int and crisis_start is not None:
        return pd.date_range(max(index.min(), crisis_start - pd.Timedelta(days=window_size)), min(index.max(), crisis_start + pd.Timedelta(days=window_size - 1)))
    elif type(window_size) == tuple and crisis_start is not None:
        return pd.date_range(max(index.min(), crisis_start - pd.Timedelta(days=window_size[0])), min(index.max(), crisis_start + pd.Timedelta(days=window_size[1] - 1)))
    else:
        return pd.date_range(index.min(), index.max())

def extract_data(filename: str, crisis_start: pd.Timestamp, num_samples: int = 100, window_size: int | Tuple[int, int] | None = (60, 30)) -> Tuple[pd.DataFrame, pd.DataFrame]:
    src_df = pd.read_excel(filename)
    
    new_cols = ['brak', 'negatywny', 'neutralny', 'pozytywny']
    new_cols_ex = [c for c in new_cols if c in src_df['Wydźwięk'].unique().tolist()]
    src_df[new_cols_ex] = pd.get_dummies(src_df['Wydźwięk'])
    for col in new_cols:
        if col not in src_df.columns:
            src_df[col] = 0

    df = src_df[['Data wydania'] + new_cols].groupby(['Data wydania']).sum()

    df = df.reindex(clip_date_range(df.index, crisis_start, window_size))
    df[new_cols] = df[new_cols].fillna(0)

    df['suma'] = df[new_cols].sum(axis=1)
    df['label'] = df.index >= crisis_start
    if np.unique(df['label']).shape[0] != 2:
        warn(f'Samples from only 1 class in {filename}.')
    if df.shape[0] == 0:
        warn(f'No data after clipping for {filename}.')

    text = src_df.apply(lambda x: ".".join([str(x['Tytuł publikacji']), str(x['Lead']), str(x['Kontekst publikacji'])]), axis=1)
    text_df = src_df[['Data wydania']].copy()
    text_df['text'] = text
    texts = []
    for date in df.index:
        daily_posts = text_df[text_df['Data wydania'] == date]
        texts.append(daily_posts if daily_posts.shape[0] <= num_samples else daily_posts.sample(n=num_samples))
    text_df = pd.concat(texts).reset_index(drop=True)
    
    return df, text_df

def load_data(filenames: Iterable[str], crisis_dates: Iterable[pd.Timestamp], num_samples: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert len(filenames) == len(crisis_dates)
    dfs, text_dfs = [], []
    for i, (fname, date) in enumerate(tqdm(zip(filenames, crisis_dates), total=len(filenames))):
        df, text_df = extract_data(fname, date, num_samples)
        df = df.reset_index(names='Data wydania')
        df['group'] = i
        text_df['group'] = i
        dfs.append(df)
        text_dfs.append(text_df)
    return pd.concat(dfs, ignore_index=True), pd.concat(text_dfs, ignore_index=True)

class DictDataset(Dataset):
    def __init__(self, items: dict) -> None:
        super().__init__()
        self.items = items
        self.len = len(self.items[list(self.items.keys())[0]])
    
    def __getitem__(self, index):
        return {key: val[index] for key, val in self.items.items()}
    
    def __len__(self) -> int:
        return self.len

class SeriesDataset(Dataset):
    def __init__(self, series: pd.Series) -> None:
        super().__init__()
        self.series = series
    
    def __getitem__(self, index):
        return self.series.iloc[index]
    
    def __len__(self) -> int:
        return self.series.shape[0]
    
class TextVectorizer(pl.LightningModule):
    def __init__(self, pretrained_name: str, max_length: int = 256, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_length = max_length

        self.model = RobertaModel.from_pretrained(pretrained_name)
        self.model._modules['pooler'] = torch.nn.Identity()
    
    def forward(self, x) -> Any:
        return self.model(**x).pooler_output.mean(dim=1)

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
    X = torch.tensor(df[numeric_cols].values, dtype=torch.float32)
    y = torch.tensor(df['label'], dtype=torch.long)
    embeddings = torch.tensor(df['embedding'], dtype=torch.float32)
    groups = df['group']

    new_features = []
    for i in groups.unique():
        X_i = X[groups == i]
        X_diff = torch.ones_like(X_i)
        X_diff[1:] = X_i[1:] / X_i[:-1]
        X_diff = torch.pow(X_diff, 2)
        X_diff = (X_diff - 1.) / (X_diff + 1)
        zeros = X_i == 0.
        X_diff[zeros & X_diff.isnan()] = 0.
        X_diff.nan_to_num_(1.)
        X_ratio = X_i[:, :-1] / X_i[:, -1:]
        X_ratio.nan_to_num_(0.)
        new_features.append(torch.cat((X_diff, X_ratio), dim=1))
        X_i[...] = torch.tensor(StandardScaler().fit_transform(X_i))
    
    X = torch.cat((X, torch.cat(new_features, dim=0), embeddings), dim=1)

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

def split_dataset(ds: Dataset, groups: torch.Tensor, n_splits: int = 10, validate: bool = True) -> Tuple[Dataset, Dataset] | Tuple[Dataset, Dataset, Dataset]:
    fold = GroupKFold(n_splits)
    splits = list(fold.split(ds, groups=groups))
    train_idx = splits[0][0]
    test_idx = splits[0][1]
    if validate:
        val_idx = splits[1][1]
        train_idx = np.array(list(set(train_idx) - set(val_idx)))
        return Subset(ds, train_idx), Subset(ds, test_idx), Subset(ds, val_idx)
    else:
        return Subset(ds, train_idx), Subset(ds, test_idx)

def fold_dataset(ds: Dataset, groups: torch.Tensor, n_splits: int = 10, validate: bool = True) -> List[Tuple[Dataset, Dataset]] | List[Tuple[Dataset, Dataset, Dataset]]:
    fold = GroupKFold(n_splits)
    splits = list(fold.split(ds, groups=groups))
    if validate:
        train_idx, test_idx = tuple(zip(*splits))
        val_idx = test_idx[1:] + test_idx[:1]
        train_idx = [np.array(list(set(t) - set(v))) for t, v in zip(train_idx, val_idx)]
        return [(Subset(ds, t), Subset(ds, t2), Subset(ds, v)) for t, v, t2 in zip(train_idx, val_idx, test_idx)]
    else:
        return [(Subset(ds, train_idx), Subset(ds, test_idx)) for train_idx, test_idx in splits]

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

def train_model(
        model: pl.LightningModule,
        train_ds: Dataset,
        val_ds: Dataset | None = None,
        batch_size: int = 512,
        num_workers: int = 10,
        verbose: bool = True,
        deterministic: bool = False
) -> pl.Trainer:
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_ds else None
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        save_top_k=1
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        logger=CSVLogger(os.getcwd()),
        callbacks=[
        checkpoint_callback,
        EarlyStopping(monitor='val_f1', mode='max', patience=20)
        ],
        max_epochs=-1,
        enable_model_summary=verbose,
        enable_progress_bar=verbose,
        deterministic=deterministic
    )

    trainer.fit(model, train_dl, val_dl)
    return trainer

def test_model(
        trainer: pl.Trainer,
        test_ds: Dataset,
        batch_size: int = 512,
        num_workers: int = 10,
        checkpoint: bool = True,
        verbose: bool = True
) -> dict:
    ckpt_path = 'best' if checkpoint else None
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return trainer.test(dataloaders=test_dl, ckpt_path=ckpt_path, verbose=verbose)[0]

def cross_validate(
        model_class: type,
        model_params: Tuple,
        ds: Dataset,
        groups: torch.Tensor,
        n_splits: int = 10,
        batch_size: int = 512,
        num_workers: int = 10,
        deterministic: bool = False
) -> pd.DataFrame:
    folds = fold_dataset(ds, groups, n_splits)
    stats = []
    for train_ds, test_ds, val_ds in tqdm(folds):

        model = model_class(*model_params)

        trainer = train_model(model, train_ds, val_ds, batch_size, num_workers, False, deterministic)
        stats.append(test_model(trainer, test_ds, batch_size, num_workers, True, False))
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

    if deterministic:
        seed_everything(42, workers=True)

    DAYS_DF_PATH = 'other_data/days_df.feather'
    POSTS_DF_PATH = 'other_data/posts_df.feather'
    EMBEDDINGS_PATH = 'other_data/embeddings.pt'

    if end_to_end or not (os.path.isfile(DAYS_DF_PATH) and os.path.isfile(POSTS_DF_PATH)):
        DATA_DIR = 'crisis_data'
        FILE_BLACKLIST = [
            'crisis_data/Jan Szyszko_Córka leśniczego.xlsx',
            'crisis_data/Komenda Główna Policji.xlsx',
            'crisis_data/Ministerstwo Zdrowia_respiratory od handlarza bronią.xlsx',
            'crisis_data/Polska Grupa Energetyczna.xlsx',
            'crisis_data/Polski Związek Kolarski.xlsx',
            'crisis_data/Zbój_energetyk.xlsx'
        ]

        crisis = pd.read_excel('crisis_data/Daty_kryzysów.xlsx').dropna()
        crisis = crisis[~crisis['Plik'].apply(lambda x: os.path.join(DATA_DIR, x) in FILE_BLACKLIST)]

        days_df, text_df = load_data(crisis['Plik'].apply(lambda x: os.path.join(DATA_DIR, x)).to_list(), crisis['Data'].to_list())
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
        model = TextVectorizer('sdadas/polish-distilroberta')
        trainer = pl.Trainer(accelerator='gpu')
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

    cross_validate(MyModel, (782, 128, 2), ds, groups, deterministic=deterministic)

if __name__ == '__main__':
    main()
