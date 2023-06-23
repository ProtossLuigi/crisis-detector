from typing import Any, Tuple
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as pl
from transformers import AutoTokenizer
from data_tools import SeriesDataset

from embedder import TextEmbedder

class PostDetector(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

def create_dataset(df: pd.DataFrame, embedder: TextEmbedder, sequence_len: int = 50, text_length: int = 256, batch_size: int = 128, window_size: int | Tuple[int, int] | None = None, deterministic: bool = False):
    if window_size is None:
        window_size = (sequence_len * 2 - 1, sequence_len)
    elif type(window_size) == int:
        window_size = (window_size + sequence_len - 1, window_size)

    times = torch.tensor(df['time'].apply(lambda x: x.timetuple()[:6]))

    tokenizer = AutoTokenizer.from_pretrained(embedder.pretrained_name)
    ds = SeriesDataset(df['text'])
    collate_fn = lambda x: tokenizer(x, truncation=True, padding=True, max_length=text_length, return_tensors='pt')
    dl = DataLoader(ds, batch_size, num_workers=10, collate_fn=collate_fn, pin_memory=True)
    trainer = pl.Trainer(devices=1, precision='bf16-mixed', logger=False, deterministic=deterministic)
    embeddings = trainer.predict(embedder, dl)
    embeddings = torch.cat(embeddings, dim=0)

    sentiments = ['brak', 'negatywny', 'neutralny', 'pozytywny']
    sentiment_data = pd.get_dummies(df['sentiment'])
    for sent in sentiments:
        if sent not in sentiment_data.columns:
            sentiment_data[sent] = False
    sentiment_data = torch.tensor(sentiment_data.values, dtype=torch.float32)

    features = torch.cat((embeddings, sentiment_data), dim=-1)
    labels = torch.tensor(df['date_label'], dtype=torch.long)
    groups = torch.tensor(df['group'].values)

    times_seq = []
    features_seq = []
    labels_seq = []

    for g in tqdm(groups.unique()):
        selector0 = ((groups == g) & ~labels).nonzero().flatten()[-window_size[0]:]
        selector1 = ((groups == g) & labels).nonzero().flatten()[:window_size[1]]
        selector = torch.zeros_like(labels, dtype=bool)
        selector[torch.cat((selector0, selector1))] = True
        times_g = times[selector]
        features_g = features[selector]
        labels_g = labels[selector]
        for i in range(labels.shape[0] - sequence_len + 1):
            ...