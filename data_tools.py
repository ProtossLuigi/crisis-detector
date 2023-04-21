from typing import Iterable, Tuple
from warnings import warn
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

def clip_date_range(index: pd.DatetimeIndex, crisis_start: pd.Timestamp | None = None, window_size: int | Tuple[int, int] | None = None) -> pd.DatetimeIndex:
    if type(window_size) == int and crisis_start is not None:
        return pd.date_range(max(index.min(), crisis_start - pd.Timedelta(days=window_size)), min(index.max(), crisis_start + pd.Timedelta(days=window_size - 1)))
    elif type(window_size) == tuple and crisis_start is not None:
        return pd.date_range(max(index.min(), crisis_start - pd.Timedelta(days=window_size[0])), min(index.max(), crisis_start + pd.Timedelta(days=window_size[1] - 1)))
    else:
        return pd.date_range(index.min(), index.max())

def extract_data(
        filename: str,
        crisis_start: pd.Timestamp | None = None,
        num_samples: int = 0,
        window_size: int | Tuple[int, int] | None = (60, 30)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    src_df = pd.read_excel(filename)
    
    new_cols = ['brak', 'negatywny', 'neutralny', 'pozytywny']
    new_cols_ex = [c for c in new_cols if c in src_df['Wydźwięk'].unique().tolist()]
    src_df[new_cols_ex] = pd.get_dummies(src_df['Wydźwięk'])
    for col in new_cols:
        if col not in src_df.columns:
            src_df[col] = 0

    if crisis_start is not None:
        src_df['label'] = src_df['Data wydania'] >= crisis_start
    else:
        
        if src_df['Kryzys'].hasnans:
            if src_df['Kryzys'].nunique(dropna=False) != 2:
                warn(f'Invalid Kryzys column values in {filename}.')
            src_df['label'] = ~src_df['Kryzys'].isna()
        else:
            src_df['Kryzys'] = src_df['Kryzys'].apply(lambda x: x[:3])
            if src_df['Kryzys'].nunique(dropna=False) != 2:
                warn(f'Invalid Kryzys column values in {filename}.')
            src_df['label'] = src_df['Kryzys'] != 'NIE'

    df = src_df[['Data wydania'] + new_cols].groupby(['Data wydania']).sum()

    df = df.reindex(clip_date_range(df.index, crisis_start, window_size))
    df[new_cols] = df[new_cols].fillna(0)

    df['suma'] = df[new_cols].sum(axis=1)
    df = df.join(src_df[['Data wydania', 'label']].groupby('Data wydania').any())
    if df['label'].hasnans:
        df['label'] = df['label'].bfill() & df['label'].ffill()

    if np.unique(df['label']).shape[0] != 2:
        warn(f'Samples from only 1 class in {filename}.')
    if df.shape[0] == 0:
        warn(f'No data after clipping for {filename}.')

    text = src_df.apply(lambda x: ".".join([str(x['Tytuł publikacji']), str(x['Lead']), str(x['Kontekst publikacji'])]), axis=1)
    text_df = src_df[['Data wydania', 'label']].copy()
    text_df['text'] = text
    texts = []
    for date in df.index:
        daily_posts = text_df[text_df['Data wydania'] == date]
        texts.append(daily_posts if num_samples == 0 or daily_posts.shape[0] <= num_samples else daily_posts.sample(n=num_samples))
    text_df = pd.concat(texts).reset_index(drop=True)
    
    return df, text_df

def load_data(filenames: Iterable[str], crisis_dates: Iterable[pd.Timestamp] | None = None, num_samples: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not crisis_dates:
        crisis_dates = [None] * len(filenames)
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
    def __init__(self, items: dict, labels: torch.Tensor | None = None) -> None:
        super().__init__()
        self.items = items
        self.labels = labels
        self.len = len(self.items[list(self.items.keys())[0]])
    
    def __getitem__(self, index):
        sample = {key: val[index] for key, val in self.items.items()}
        if self.labels is None:
            return sample
        else:
            return sample, self.labels[index]
    
    def __len__(self) -> int:
        return self.len

class SeriesDataset(Dataset):
    def __init__(self, series: pd.Series, labels: torch.Tensor | None = None) -> None:
        super().__init__()
        self.series = series
        self.labels = labels
    
    def __getitem__(self, index):
        if self.labels is None:
            return self.series.iloc[index]
        else:
            return self.series.iloc[index], self.labels[index]
    
    def __len__(self) -> int:
        return self.series.shape[0]
    