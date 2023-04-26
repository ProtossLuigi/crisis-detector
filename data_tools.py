from typing import Iterable, Tuple, List
from warnings import warn
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

import torch
from torch.utils.data import Dataset

DATA_DIR = 'dane'
VERIFIED_DIR = 'dane/Etap I - zweryfikowane szeregi'
DATES_FILE = 'dane/Daty_kryzysów.xlsx'

FILE_BLACKLIST = [
    'Daty_kryzysów.xlsx',
    'Crisis Detector - lista wątków_.docx',
    'Fake news_baza publikacji.xlsx'
]

def get_verified_data() -> List[str]:
    filenames = [os.path.join(VERIFIED_DIR, file) for file in os.listdir(VERIFIED_DIR) if os.path.isfile(os.path.join(VERIFIED_DIR, file))]
    filenames = [fname for fname in filenames if os.path.basename(fname) not in FILE_BLACKLIST]
    return filenames

def get_all_data() -> List[str]:
    filenames = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, file))]
    filenames += [os.path.join(VERIFIED_DIR, file) for file in os.listdir(VERIFIED_DIR) if os.path.isfile(os.path.join(VERIFIED_DIR, file))]
    filenames = [fname for fname in filenames if os.path.basename(fname).replace("'","_") not in FILE_BLACKLIST]
    return filenames

def get_crisis_dates() -> pd.DataFrame:
    return pd.read_excel(DATES_FILE)

def get_data_with_dates(files: List[str]) -> pd.DataFrame:
    fnames = list(map(os.path.basename, files))
    dates = get_crisis_dates()
    dates1 = dates[dates['Plik'].isin(fnames)]
    dates2 = dates[dates['Plik2'].isin(fnames)]
    if len(dates1) > len(dates2):
        dates1['path'] = dates1['Plik'].apply(lambda x: files[fnames.index(x)])
        dates = dates1
    else:
        dates2['path'] = dates2['Plik2'].apply(lambda x: files[fnames.index(x)])
        dates = dates2
    return dates[['path', 'Data']]

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
        window_size: int | Tuple[int, int] | None = (60, 30),
        drop_invalid: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame] | None:
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
                if drop_invalid:
                    return None
                else:
                    warn(f'Invalid Kryzys column values in {filename}.')
            src_df['label'] = ~src_df['Kryzys'].isna()
        else:
            src_df['Kryzys'] = src_df['Kryzys'].apply(lambda x: x[:3])
            if src_df['Kryzys'].nunique(dropna=False) != 2:
                if drop_invalid:
                    return None
                else:
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
        if drop_invalid:
            return None
        else:
            warn(f'Samples from only 1 class in {filename}.')
    if df.shape[0] == 0:
        if drop_invalid:
            return None
        else:
            warn(f'No data after clipping for {filename}.')

    text = src_df.apply(lambda x: " . ".join([str(x['Tytuł publikacji']), str(x['Lead']), str(x['Kontekst publikacji'])]), axis=1)
    text_df = src_df[['Data wydania', 'label']].copy()
    text_df['text'] = text
    texts = []
    for date in df.index:
        daily_posts = text_df[text_df['Data wydania'] == date]
        texts.append(daily_posts if num_samples == 0 or daily_posts.shape[0] <= num_samples else daily_posts.sample(n=num_samples))
    text_df = pd.concat(texts).reset_index(drop=True)
    
    return df, text_df

def load_data(
        filenames: Iterable[str], crisis_dates: Iterable[pd.Timestamp] | None = None, num_samples: int = 0, drop_invalid: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not crisis_dates:
        crisis_dates = [None] * len(filenames)
    assert len(filenames) == len(crisis_dates)
    dfs, text_dfs = [], []
    for i, (fname, date) in enumerate(tqdm(zip(filenames, crisis_dates), total=len(filenames))):
        dfp = extract_data(fname, date, num_samples, drop_invalid=drop_invalid)
        if dfp is None:
            continue
        df, text_df = dfp
        df = df.reset_index(names='Data wydania')
        df['group'] = i
        text_df['group'] = i
        dfs.append(df)
        text_dfs.append(text_df)
    return pd.concat(dfs, ignore_index=True), pd.concat(text_dfs, ignore_index=True)

def extract_text_data(
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
    text_df = pd.DataFrame({'text': text, 'label': labels})
    sample_size = text_df['label'].value_counts().min()
    text_df = pd.concat((text_df[text_df['label']].sample(sample_size), text_df[~text_df['label']].sample(sample_size))).sort_index().reset_index(drop=True)
    
    return text_df

def load_text_data(filenames: Iterable[str], crisis_dates: Iterable[pd.Timestamp], drop_invalid: bool = False) -> pd.DataFrame:
    assert len(filenames) == len(crisis_dates)
    dfs = []
    for i, (fname, date) in enumerate(tqdm(zip(filenames, crisis_dates), total=len(filenames))):
        df = extract_text_data(fname, date, drop_invalid=drop_invalid)
        if df is None:
            continue
        df['group'] = i
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

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
    