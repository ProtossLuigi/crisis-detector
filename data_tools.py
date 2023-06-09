from typing import Any, Iterable, Tuple, List
from warnings import warn
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DATA_DIR = 'dane'
DATA_DIR2 = 'data'
VERIFIED1_DIR = 'data/Etap I - bazy/Etap I - zweryfikowane szeregi'
VERIFIED2_DIR = 'data/Etap I - bazy/Bazy - wersja ostateczna'
FULL_TEXT_DIR = 'data/Etap I - bazy/Bazy - pełne treści publikacji'
DATES_FILE = 'data/Crisis Detector.feather'
FOLD_FILE = 'saved_objects/folds.feather'

FILE_BLACKLIST = []

def prepare_data(force: bool = False):
    if os.path.isdir(DATA_DIR2):
        if force:
            shutil.rmtree(DATA_DIR2)
        else:
            print(f"{DATA_DIR2} dir already present and force=False. Skipping...")
            return
    filepaths = [Path(dirpath, filename) for dirpath, _, filenames in os.walk(DATA_DIR) for filename in filenames]
    for filepath in tqdm(filepaths):
        if filepath.with_suffix('').name in FILE_BLACKLIST or filepath.suffix != '.xlsx':
            continue
        target_path = Path(DATA_DIR2, *filepath.with_suffix('.feather').parts[1:])
        os.makedirs(target_path.parent, exist_ok=True)
        try:
            df = pd.read_excel(filepath)
            df.columns = df.columns.astype(str)
            if 'Kryzys' in df.columns:
                df['Kryzys'] = df['Kryzys'].astype(str)
            df.to_feather(target_path)
        except Exception as e:
            print(f"Error in {filepath}")
            raise e
    if os.path.isfile('dane/Crisis Detector.xlsx'):
        df = pd.read_excel('dane/Crisis Detector.xlsx')
        df['Nazwa pliku'] = df['Nazwa pliku'].apply(lambda x: x[:-5] + '.feather' if type(x) == str else x)
        df['Nazwa pliku 2'] = df['Nazwa pliku 2'].apply(lambda x: x[:-5] + '.feather' if type(x) == str else x)
        df.to_feather(DATES_FILE)

def generate_splits(i: int = 1, n_splits: int = 5, validate: bool = True) -> None:
    topics_df = get_data_with_dates(get_verified_data(i))
    topic_data = []
    for idx, row in tqdm(enumerate(topics_df.itertuples()), desc='Loading files', total=len(topics_df)):
        df = pd.read_feather(row.path)
        magnitude = len(clip_date_range(pd.DatetimeIndex(df['Data wydania']), row.crisis_start, (59, 30)))
        topic_data.append(pd.DataFrame({'name': [row.name] * magnitude, 'id': [idx] * magnitude}))
    topic_data = pd.concat(topic_data, ignore_index=True)
    splits = GroupKFold(n_splits).split(topic_data, groups=topic_data['id'])
    if validate:
        train_idx, test_idx = tuple(zip(*splits))
        val_idx = test_idx[1:] + test_idx[:1]
        train_idx = [np.array(list(set(t) - set(v))) for t, v in zip(train_idx, val_idx)]
        splits = list(zip(train_idx, test_idx, val_idx))
    split_df = pd.DataFrame(columns=topics_df['name'])
    for fold in tqdm(splits, desc='Saving splits'):
        row = {}
        for i, split in enumerate(fold):
            for name in np.unique(topic_data['name'][split]):
                row[name] = [['train', 'test', 'val'][i]]
        split_df = pd.concat((split_df, pd.DataFrame(row)), ignore_index=True)
    split_df.to_feather(FOLD_FILE)

def get_verified_data(i: int = 1) -> List[str]:
    if i == 1:
        dirpath = VERIFIED1_DIR
    elif i == 2:
        dirpath = VERIFIED2_DIR
    else:
        raise ValueError('bruh')
    filenames = [Path(dirpath, file) for file in os.listdir(dirpath) if os.path.isfile(Path(dirpath, file))]
    filenames = [fname for fname in filenames if fname.with_suffix('').name not in FILE_BLACKLIST]
    return filenames

def get_all_data() -> List[str]:
    filenames = [Path(VERIFIED1_DIR, file) for file in os.listdir(VERIFIED1_DIR) if os.path.isfile(Path(VERIFIED1_DIR, file))]
    filenames += [Path(VERIFIED2_DIR, file) for file in os.listdir(VERIFIED2_DIR) if os.path.isfile(Path(VERIFIED2_DIR, file))]
    filenames = [fname for fname in filenames if fname.with_suffix('').name.replace("'","_") not in FILE_BLACKLIST]
    return filenames

def get_full_text_data() -> List[str]:
    filenames = [Path(FULL_TEXT_DIR, file) for file in os.listdir(FULL_TEXT_DIR) if os.path.isfile(Path(FULL_TEXT_DIR, file))]
    filenames = [fname for fname in filenames if fname.with_suffix('').name.replace("'","_") not in FILE_BLACKLIST]
    return filenames

def get_crisis_metadata(drop_nan_dates: bool = False) -> pd.DataFrame:
    df = pd.read_feather(DATES_FILE)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    elif '' in df.columns:
        df = df.drop(columns=[''])
    if len(df['Baza'].astype(bool).unique()) == 1:
        df['Baza'] = df['Baza'] == 'TRUE'
    df = df[df['Baza']].drop(columns=['Baza'])
    if drop_nan_dates:
        df = df[df['Data wybuchu kryzysu'].notna()].reset_index(drop=True)
    return df

def get_data_with_dates(files: List[str], drop_nan_dates: bool = True) -> pd.DataFrame:
    fnames = list(map(os.path.basename, files))
    metadata = get_crisis_metadata(drop_nan_dates)
    data1 = metadata[metadata['Nazwa pliku'].isin(fnames)]
    data2 = metadata[metadata['Nazwa pliku 2'].isin(fnames)]
    if len(data1) > len(data2):
        data1['path'] = data1['Nazwa pliku'].apply(lambda x: files[fnames.index(x)])
        data = data1
    else:
        data2['path'] = data2['Nazwa pliku 2'].apply(lambda x: files[fnames.index(x)])
        data = data2
    data = data.rename(columns={'Opis': 'name', 'Data wybuchu kryzysu': 'crisis_start'})
    return data[['name', 'path', 'crisis_start']]

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
        window_size: int | Tuple[int, int] | None = (59, 30),
        drop_invalid: bool = False,
        class_balance: float | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame] | None:
    src_df = pd.read_feather(filename).sort_values('Data wydania', ignore_index=True)
    
    new_cols = ['brak', 'negatywny', 'neutralny', 'pozytywny']
    sent_col = 'Wydźwięk' if 'Wydźwięk' in src_df.columns else 'Sentyment'
    text_col = 'Kontekst publikacji' if 'Kontekst publikacji' in src_df.columns else 'OCR'
    new_cols_ex = [c for c in new_cols if c in src_df[sent_col].unique().tolist()]
    src_df[new_cols_ex] = pd.get_dummies(src_df[sent_col])
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
        if crisis_start is None:
            df['label'] = df['label'].bfill() & df['label'].ffill()
        else:
            df['label'] = df.index >= crisis_start

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

    text = src_df.apply(lambda x: " . ".join([str(x['Tytuł publikacji']), str(x['Lead']), str(x[text_col])]), axis=1)
    text_df = src_df[['Data wydania', 'label']].copy()
    text_df['text'] = text
    statistic_cols = ['Ave szacunkowe (PLN)', 'Dotarcie (kontaktów)', 'Zasięg (egz.), (słuchaczy), (widzów), (UU), (subskrybentów), (obserwujących)', 'Wpływ']
    text_df['statistics'] = list(StandardScaler().fit_transform(src_df[statistic_cols].fillna(0).values))
    texts = []
    for date in df.index:
        daily_posts = text_df[text_df['Data wydania'] == date]
        if class_balance is not None and daily_posts['label'].any():
            total_samples = len(daily_posts)
            pos_samples = sum(daily_posts['label'])
            neg_samples = sum(~daily_posts['label'])
            ratio = min(total_samples / num_samples, pos_samples / (num_samples * class_balance), neg_samples / (num_samples * (1. - class_balance)), 1.)
            if ratio == 0.:
                continue
            # idx_0 = np.argsort(daily_posts[~daily_posts['label']]['statistics'].apply(np.sum))[-int(num_samples * ratio * (1. - class_balance)):]
            # idx_1 = np.argsort(daily_posts[daily_posts['label']]['statistics'].apply(np.sum))[-int(num_samples * ratio * class_balance):]
            # posts_0 = daily_posts[~daily_posts['label']].iloc[idx_0]
            # posts_1 = daily_posts[daily_posts['label']].iloc[idx_1]
            posts_1 = daily_posts[daily_posts['label']].sample(n=int(num_samples * ratio * class_balance))
            posts_0 = daily_posts[~daily_posts['label']].sample(n=int(num_samples * ratio * (1. - class_balance)))
            texts.append((posts_0 + posts_1).sort_index())
        else:
            texts.append(daily_posts if num_samples == 0 or daily_posts.shape[0] <= num_samples else daily_posts.sample(n=num_samples))
    text_df = pd.concat(texts).reset_index(names='id')
    
    return df, text_df

def load_data(
        metadata: pd.DataFrame, num_samples: int = 0, drop_invalid: bool = False, class_balance: float | None = None, window_size: int | Tuple[int, int] | None = (59, 30)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if 'crisis_start' not in metadata.columns:
        metadata['crisis_start'] = None
    dfs, text_dfs = [], []
    for i, row in enumerate(tqdm(metadata.itertuples(), total=len(metadata))):
        try:
            dfp = extract_data(row.path, row.crisis_start, num_samples, drop_invalid=drop_invalid, class_balance=class_balance, window_size=window_size)
            if dfp is None:
                continue
        except KeyError as e:
            warn(f'Missing column {e.args[0]} in {row.path}. Skipping...')
            continue
        df, text_df = dfp
        df = df.reset_index(names='Data wydania')
        df['group'] = i
        df['name'] = row.name
        text_df['group'] = i
        text_df['name'] = row.name
        dfs.append(df)
        text_dfs.append(text_df)
    return pd.concat(dfs, ignore_index=True), pd.concat(text_dfs, ignore_index=True)

def extract_text_data(
        filename: str,
        crisis_start: pd.Timestamp,
        window_size: int | Tuple[int, int] = 30,
        samples_limit: int | None = None,
        drop_invalid: bool = False
) -> pd.DataFrame | None:
    src_df = pd.read_feather(filename)

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
    text_df = pd.DataFrame({'text': text, 'label': labels, 'time': src_df['Data wydania'], 'sentiment': src_df['Wydźwięk']})
    # text_df['impact'] = src_df['Dotarcie (kontaktów)'] > np.quantile(src_df['Dotarcie (kontaktów)'], .5)
    statistic_cols = ['Ave szacunkowe (PLN)', 'Dotarcie (kontaktów)', 'Zasięg (egz.), (słuchaczy), (widzów), (UU), (subskrybentów), (obserwujących)', 'Wpływ']
    text_df['statistics'] = list(src_df[statistic_cols].fillna(0.).values)
    text_df['impact'] = src_df['Dotarcie (kontaktów)']
    counts = text_df['label'].value_counts()
    if len(counts) < 2:
        sample_size = 0
    else:
        sample_size = counts.min()
    if samples_limit is not None:
        sample_size = min(sample_size, samples_limit // 2)
    text_df = pd.concat((text_df[text_df['label']].sample(sample_size), text_df[~text_df['label']].sample(sample_size))).sort_index(ignore_index=True)
    # if sample_size > 0:
    #     idx_0 = np.argsort(text_df[~text_df['label']]['statistics'].apply(np.sum))[-sample_size:]
    #     idx_1 = np.argsort(text_df[text_df['label']]['statistics'].apply(np.sum))[-sample_size:]
    # else:
    #     idx_0, idx_1 = [], []
    # text_df = pd.concat((text_df[~text_df['label']].iloc[idx_0], text_df[text_df['label']].iloc[idx_1])).sort_index(ignore_index=True)
    text_df['time_label'] = text_df['time'] >= crisis_start
    
    return text_df.sort_values(by='time', ignore_index=True)

def load_text_data(filenames: Iterable[str], crisis_dates: Iterable[pd.Timestamp], samples_limit: int | None = None, drop_invalid: bool = False) -> pd.DataFrame:
    assert len(filenames) == len(crisis_dates)
    dfs = []
    for i, (fname, date) in enumerate(tqdm(zip(filenames, crisis_dates), total=len(filenames))):
        try:
            df = extract_text_data(fname, date, samples_limit=samples_limit, drop_invalid=drop_invalid)
            if df is None:
                continue
            df['group'] = i
            dfs.append(df)
        except KeyError as e:
            warn(f'Missing column {e.args[0]} in {fname}. Skipping...')
    return pd.concat(dfs, ignore_index=True)

class DictDataset(Dataset):
    def __init__(self, items: dict, labels: torch.Tensor | None = None) -> None:
        super().__init__()
        self.items = items
        self.items['label'] = labels
        self.len = len(self.items[list(self.items.keys())[0]])
    
    def __getitem__(self, index):
        sample = {key: val[index] for key, val in self.items.items()}
        return sample
    
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

class SimpleDataset(Dataset):
    def __init__(self, *data) -> None:
        super().__init__()
        if len(data) == 1:
            self.data = data
        else:
            self.data = list(zip(*data))
    
    def __getitem__(self, index) -> Any:
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)
