from typing import Any, List, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from itertools import compress
import json

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from lightning.pytorch import seed_everything
import torchmetrics
from transformers import AutoTokenizer

from data_tools import get_all_data, get_data_with_dates, load_data
from embedder import TextEmbedder
from aggregator import EmbeddingAggregator
from crisis_detector import MyClassifier, ShiftMetric, Shift2Metric

torch.set_float32_matmul_precision('high')

class CombinedDataset(Dataset):
    def __init__(self, day_features: torch.Tensor, tokens: List[dict], sections: List[List[int]], labels: torch.Tensor) -> None:
        super().__init__()

        self.day_features = day_features
        self.tokens = tokens
        self.sections = sections
        self.labels = labels
    
    def __getitem__(self, index) -> Any:
        return self.day_features[index], self.tokens[index], self.sections[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

class CrisisDetector(pl.LightningModule):
    def __init__(
            self,
            embedder: TextEmbedder,
            aggregator: EmbeddingAggregator,
            detector: MyClassifier,
            lr: float = 1e-5,
            weight_decay: float = 0.01,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.weight_decay = weight_decay

        self.embedder = embedder
        self.aggregator = aggregator
        self.detector = detector

        self.f1 = torchmetrics.F1Score('multiclass', num_classes=2, average=None)
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=2, average='micro')
        self.prec = torchmetrics.Precision('multiclass', num_classes=2, average=None)
        self.rec = torchmetrics.Recall('multiclass', num_classes=2, average=None)
        self.shift = ShiftMetric()
        self.shift2 = Shift2Metric()
    
    def init_loss_fn(self, class_ratios: torch.Tensor | None = None) -> None:
        self.class_weights = .5 / class_ratios if class_ratios else None
        self.loss_fn = nn.CrossEntropyLoss(self.class_weights)
    
    def forward(self, x_features: torch.Tensor, x_tokens: dict) -> Any:
        embeddings = self.embedder(x_tokens).hidden_states[0][:, 0, :]
        embeddings = self.aggregator(embeddings)
        return self.detector(torch.cat((x_features, embeddings), dim=-1))
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(*X)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(*X)
        self.acc.update(torch.argmax(y_pred, -1), y)
        self.f1.update(torch.argmax(y_pred, -1), y)
        self.log('val_loss', self.loss_fn(y_pred, y), on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        metrics = {
            'val_acc': self.acc.compute(),
            'val_f1': self.f1.compute().mean()
        }
        self.log_dict(metrics)
        self.acc.reset()
        self.f1.reset()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(*X)
        if self.print_test_samples:
            print(torch.argmax(y_pred, -1) - y)
        self.acc.update(torch.argmax(y_pred, -1), y)
        self.f1.update(torch.argmax(y_pred, -1), y)
        self.prec.update(torch.argmax(y_pred, -1), y)
        self.rec.update(torch.argmax(y_pred, -1), y)
        self.shift.update(torch.argmax(y_pred, -1), y)
        self.shift2.update(torch.argmax(y_pred, -1), y)
        self.log('test_loss', self.loss_fn(y_pred, y), on_epoch=True)
    
    def on_test_epoch_end(self) -> None:
        metrics = {
            'test_acc': self.acc.compute()
        }
        metrics['test_f1_neg'], metrics['test_f1_pos'] = self.f1.compute()
        metrics['test_precision_neg'], metrics['test_precision_pos'] = self.prec.compute()
        metrics['test_recall_neg'], metrics['test_recall_pos'] = self.rec.compute()
        shift = self.shift.compute().float()
        metrics['test_shift'], metrics['test_shift_std'] = shift.mean(), shift.std()
        shift = self.shift2.compute().float()
        metrics['test_shift2'], metrics['test_shift2_std'] = shift.mean(), shift.std()
        self.log_dict(metrics)
        self.acc.reset()
        self.f1.reset()
        self.prec.reset()
        self.rec.reset()
        self.shift.reset()
        self.shift2.reset()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, 'min', .5, 10),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

def tokenize_dataset(text: List[str], tokenizer_name: str = 'xlm-roberta-base', batch_size: int | None = None, max_length: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenize_fn = lambda x: tokenizer(x, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    if batch_size:
        input_ids, attention_mask = [], []
        for i in tqdm(range(0, len(text), batch_size)):
            for key, val in tokenize_fn(text[i:i+batch_size]).items():
                if key == 'input_ids':
                    input_ids.append(val)
                elif key == 'attention_mask':
                    attention_mask.append(val)
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
    else:
        tokens = tokenize_fn(text)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
    return input_ids, attention_mask

def get_day_splits(days_df: pd.DataFrame, text_df: pd.DataFrame) -> List[int]:
    combined_df = pd.merge(days_df[['group', 'Data wydania']], text_df, how='left', on=['group', 'Data wydania'])
    empty_days = combined_df[combined_df['text'].isna()].index
    sections = [0] + np.cumsum(text_df.groupby(['group', 'Data wydania']).count()['text']).tolist()
    #TODO
    return sections

def get_day_features(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, pd.Series]:
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
    return X, y, groups

def create_dataset(
        features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sections: List[int],
        labels: torch.Tensor,
        groups: pd.Series,
        sequence_len: int = 30
) -> Dataset:
    seq_features = []
    seq_input_ids = []
    seq_attention_mask = []
    seq_sections = []
    seq_labels = []
    seq_groups = []
    for g in groups.unique():
        selector = groups == g
        features_g = features[selector]
        input_ids_g = input_ids[selector]
        attention_mask_g = attention_mask[selector]
        labels_g = labels[selector]



        # TODO

def main():
    deterministic = True
    end_to_end = False
    text_samples = 50

    if deterministic:
        seed_everything(42, workers=True)
    
    DAYS_DF_PATH = 'saved_objects/days_df.feather'
    POSTS_DF_PATH = 'saved_objects/posts_df' + str(text_samples) + '.feather'
    TOKENS_PATH = 'saved_objects/tokens' + str(text_samples) + '.json'

    if end_to_end or not (os.path.isfile(DAYS_DF_PATH) and os.path.isfile(POSTS_DF_PATH)):

        data = get_data_with_dates(get_all_data())

        days_df, text_df = load_data(data['path'].to_list(), data['crisis_start'].to_list(), text_samples, True)
        days_df.to_feather(DAYS_DF_PATH)
        text_df.to_feather(POSTS_DF_PATH)

        if deterministic:
            seed_everything(42, workers=True)
    else:
        days_df = pd.read_feather(DAYS_DF_PATH)
        text_df = pd.read_feather(POSTS_DF_PATH)
    
    if end_to_end or not os.path.isfile(TOKENS_PATH):
        tokens = tokenize_dataset(text_df['text'].to_list(), batch_size=256)

        with open(TOKENS_PATH, 'w') as f:
            json.dump(tokens, f)
        
        if deterministic:
            seed_everything(42, workers=True)
    else:
        with open(TOKENS_PATH, 'r') as f:
            tokens = json.load(f)
    


if __name__ == '__main__':
    main()
