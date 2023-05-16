from typing import Any, List
import os
import pandas as pd
from tqdm import tqdm
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
    def __init__(self, day_features: torch.Tensor, tokens: dict, indices: torch.Tensor, labels: torch.Tensor) -> None:
        super().__init__()

        self.day_features = day_features
        self.tokens = tokens
        self.indices = indices
        self.labels = labels
        
        assert len(self.indices) == len(self.day_features) + 1
    
    def __getitem__(self, index) -> Any:
        return (self.day_features[index], {key: val[self.indices[index]:self.indices[index+1]] for key, val in self.tokens.items()}), self.labels[index]

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

def tokenize_dataset(text: List[str], tokenizer_name: str = 'xlm-roberta-base', batch_size: int | None = None, max_length: int = 256) -> dict:
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    tokenize_fn = lambda x: tokenizer(x, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    if batch_size:
        tokens = {'input_ids': [], 'attention_mask': []}
        for i in tqdm(range(0, len(text), batch_size)):
            for key, val in tokenize_fn(text[i:i+batch_size]).items():
                tokens[key].append(val)
        for key, val in tokens.items():
            tokens[key] = torch.cat(val, dim=0)
    else:
        tokens = tokenize_fn(text)
    return tokens

def create_dataset(days_df: pd.DataFrame, text_df: pd.DataFrame, tokens: dict):
    ... #TODO

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
