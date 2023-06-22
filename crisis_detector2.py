from typing import Any
import pandas as pd

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

def create_dataset(df: pd.DataFrame, embedder: TextEmbedder, text_length: int = 256, batch_size: int = 128, deterministic: bool = False):
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

    return TensorDataset()