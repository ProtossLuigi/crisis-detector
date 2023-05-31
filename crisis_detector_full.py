from typing import Any, Iterable, List, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import lightning as pl
from lightning.pytorch import seed_everything
import torchmetrics
from transformers import AutoTokenizer

from data_tools import get_all_data, get_data_with_dates, load_data
from training_tools import init_trainer, split_dataset, fold_dataset
from embedder import TextEmbedder
from aggregator import EmbeddingAggregator, MeanAggregator
from crisis_detector import MyClassifier, ShiftMetric, Shift2Metric, MyTransformer, TopicSampler

torch.set_float32_matmul_precision('high')

def merge_indices(*tensors) -> torch.Tensor:
    return torch.cat(tensors).unique()

class CombinedDataset(Dataset):
    def __init__(
            self,
            day_features: torch.Tensor,
            post_features: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            sequences: torch.Tensor,
            labels: torch.Tensor,
            day_posts: List[torch.Tensor],
    ) -> None:
        super().__init__()

        self.day_features = day_features
        self.post_features = post_features
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.sequences = sequences
        self.day_posts = day_posts
        self.labels = labels

        self.month_posts = [torch.cat([self.day_posts[i] for i in indexes]) for indexes in self.sequences]
        self.post_counts = torch.tensor([len(x) for x in self.day_posts])

        assert len(self.input_ids) == len(self.attention_mask)
        if self.post_features is None:
            self.post_features = torch.empty((self.input_ids.shape[0], 0), dtype=torch.float32)
    
    def __getitem__(self, index) -> Any:
        if isinstance(index, int):
            day_indices = self.sequences[index]
            post_indices = self.month_posts[index]
            return (
                self.day_features[day_indices],
                self.post_features[post_indices],
                self.input_ids[post_indices],
                self.attention_mask[post_indices],
                self.post_counts[day_indices]
            ), self.labels[index]
        else:
            day_indices = self.sequences[index]
            post_indices = torch.cat([self.month_posts[i] for i in index])
            return (
                self.day_features[day_indices],
                self.post_features[post_indices],
                self.input_ids[post_indices],
                self.attention_mask[post_indices],
                self.post_counts[day_indices].flatten()
            ), self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)
    
    @staticmethod
    def collate(batch):
        X, y = tuple(zip(*batch))
        X = tuple(zip(*X))
        return (
            torch.stack(X[0]),
            torch.cat(X[1]),
            torch.cat(X[2]), 
            torch.cat(X[3]), 
            torch.cat(X[4])
        ), torch.stack(y)

class CrisisDetector(pl.LightningModule):
    def __init__(
            self,
            embedder: TextEmbedder,
            aggregator: EmbeddingAggregator,
            detector: MyClassifier,
            embedder_batch_size: int = 0,
            lr: float = 1e-5,
            weight_decay: float = 0.01,
            print_test_samples: bool = False,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.batch_size = embedder_batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.print_test_samples = print_test_samples

        self.embedder = embedder
        self.aggregator = aggregator
        self.detector = detector

        self.f1 = torchmetrics.F1Score('multiclass', num_classes=2, average=None)
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=2, average='micro')
        self.prec = torchmetrics.Precision('multiclass', num_classes=2, average=None)
        self.rec = torchmetrics.Recall('multiclass', num_classes=2, average=None)
        self.shift = ShiftMetric()
        self.shift2 = Shift2Metric()
        
        self.init_loss_fn()
    
    def init_loss_fn(self, class_ratios: torch.Tensor | None = None) -> None:
        self.class_weights = .5 / class_ratios if class_ratios else None
        self.loss_fn = nn.CrossEntropyLoss(self.class_weights)
    
    def forward(self, day_features: torch.Tensor, post_features: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, post_counts: torch.Tensor) -> Any:
        if self.batch_size > 0:
            embeddings = []
            for i in range(0, input_ids.shape[0], self.batch_size):
                embeddings.append(self.embedder(input_ids=input_ids[i:i+self.batch_size], attention_mask=attention_mask[i:i+self.batch_size]).hidden_states[0][:, 0, :])
            if embeddings:
                embeddings = torch.cat(embeddings)
            else:
                embeddings = torch.zeros((0, self.embedder.model.config.hidden_size), dtype=torch.float32, device=self.device)
        else:
            embeddings = self.embedder(input_ids=input_ids, attention_mask=attention_mask).hidden_states[0][:, 0, :]
        post_features = torch.cat((embeddings, post_features), dim=1)
        post_features = pad_sequence(torch.split(post_features, post_counts), batch_first=True)
        if post_features.shape[1] < self.aggregator.sample_size:
            post_features = nn.functional.pad(post_features, (0, 0, 0, self.aggregator.sample_size - post_features.shape[1]))
        post_features = self.aggregator(post_features)
        post_features = post_features.view(day_features.shape[0], day_features.shape[1], post_features.shape[1])
        return self.detector(torch.cat((day_features, post_features), dim=-1))
    
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

def get_day_splits(days_df: pd.DataFrame, text_df: pd.DataFrame) -> torch.Tensor:
    combined_df = pd.merge(days_df[['group', 'Data wydania']], text_df, how='left', on=['group', 'Data wydania'])
    sections = combined_df.groupby(['group', 'Data wydania'], dropna=False)['text'].count()
    return torch.tensor(sections.values)

def get_day_posts(days_df: pd.DataFrame, text_df: pd.DataFrame) -> torch.Tensor:
    splits = get_day_splits(days_df, text_df)
    splits = torch.cumsum(torch.cat(torch.tensor([0]), splits))
    return [list(range(splits[i], splits[i+1])) for i in range(len(splits)-1)]

def get_day_features(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, pd.Series]:
    numeric_cols = ['brak', 'pozytywny', 'neutralny', 'negatywny', 'suma']
    X = torch.tensor(df[numeric_cols].values, dtype=torch.long)
    y = torch.tensor(df['label'], dtype=torch.long)
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
    
    X = torch.cat(new_features, dim=0).to(torch.float32)
    return X, y, groups

def create_dataset(
        day_features: torch.Tensor,
        post_features: torch.Tensor | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        post_counts: List[int],
        labels: torch.Tensor,
        day_groups: pd.Series,
        post_groups: pd.Series,
        sequence_len: int = 30
) -> Tuple[Dataset, torch.Tensor]:
    if post_features is None:
        post_features = torch.empty((input_ids.shape[0], 0), dtype=torch.float32)
    if type(post_counts) != list:
        post_counts = list(post_counts)
    seq_day_features = []
    daily_posts = [0] + np.cumsum(post_counts).tolist()
    daily_posts = [list(range(daily_posts[i], daily_posts[i+1])) for i in range(len(daily_posts) - 1)]
    seq_daily_posts = []
    seq_post_counts = []
    seq_labels = []
    seq_groups = []
    print('Creating dataset...')
    for g in tqdm(day_groups.unique()):
        day_selector = day_groups == g
        post_selector = post_groups == g
        features_g = features[day_selector]
        daily_posts_g = [daily_posts[i] for i in day_selector.index[day_selector]]
        post_counts_g = [post_counts[i] for i in day_selector.index[day_selector]]
        labels_g = labels[day_selector]
        for i in range(sum(day_selector) - sequence_len + 1):
            seq_day_features.append(features_g[i:i+sequence_len])
            seq_daily_posts.append([id for post_ids in daily_posts_g[i:i+sequence_len] for id in post_ids])
            seq_post_counts.append([x for x in post_counts_g[i:i+sequence_len]])
            seq_labels.append(labels_g[i+sequence_len-1])
            seq_groups.append(g)
    return CombinedDataset(
        torch.stack(seq_day_features),
        post_features,
        input_ids,
        attention_mask,
        seq_daily_posts,
        seq_post_counts,
        torch.stack(seq_labels)
    ), torch.tensor(seq_groups)

def train_test(
        model: CrisisDetector, 
        ds: CombinedDataset, 
        groups: torch.Tensor, 
        batch_size: int = 1, 
        precision: str = 'bf16-mixed', 
        max_time: Any | None = None, 
        deterministic: bool = False
):
    trainer = init_trainer(precision, early_stopping=False, max_time=max_time, deterministic=deterministic)
    train_ds, test_ds = split_dataset(ds, groups, n_splits=10, validate=False)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=10, collate_fn=CombinedDataset.collate, pin_memory=True)
    # val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=10, collate_fn=CombinedDataset.collate, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_sampler=TopicSampler(SequentialSampler(test_ds)), num_workers=10, collate_fn=CombinedDataset.collate, pin_memory=True)
    trainer.fit(model, train_dl)
    trainer.test(model, test_dl, 'best')

def main():
    deterministic = True
    end_to_end = False
    text_samples = 50

    if deterministic:
        seed_everything(42, workers=True)
    
    DAYS_DF_PATH = 'saved_objects/days_df.feather'
    POSTS_DF_PATH = 'saved_objects/posts_df' + str(text_samples) + '.feather'
    INPUT_IDS_PATH = 'saved_objects/input_ids' + str(text_samples) + '.json'
    ATTENTION_MASK_PATH = 'saved_objects/attention_mask' + str(text_samples) + '.json'

    if end_to_end or not (os.path.isfile(DAYS_DF_PATH) and os.path.isfile(POSTS_DF_PATH)):

        data = get_data_with_dates(get_all_data())

        days_df, text_df = load_data(data, text_samples, True)
        days_df.to_feather(DAYS_DF_PATH)
        text_df.to_feather(POSTS_DF_PATH)

        if deterministic:
            seed_everything(42, workers=True)
    else:
        days_df = pd.read_feather(DAYS_DF_PATH)
        text_df = pd.read_feather(POSTS_DF_PATH)
    
    if end_to_end or not (os.path.isfile(INPUT_IDS_PATH) and os.path.isfile(ATTENTION_MASK_PATH)):
        input_ids, attention_mask = tokenize_dataset(text_df['text'].to_list(), batch_size=256)

        torch.save(input_ids, INPUT_IDS_PATH)
        torch.save(attention_mask, ATTENTION_MASK_PATH)
        
        if deterministic:
            seed_everything(42, workers=True)
    else:
        input_ids = torch.load(INPUT_IDS_PATH)
        attention_mask = torch.load(ATTENTION_MASK_PATH)
    
    day_features, labels, groups = get_day_features(days_df)
    ds, groups = create_dataset(
        day_features,
        input_ids,
        None,
        attention_mask,
        get_day_splits(days_df, text_df).tolist(),
        labels,
        groups,
        text_df['group']
    )
    print(ds[:])
    return
    embedder = TextEmbedder('xlm-roberta-base')
    post_features = embedder.model.config.hidden_size + ds[0][0][1].shape[-1]
    aggregator = MeanAggregator(text_samples, post_features)
    detector = MyTransformer(input_dim=post_features + ds[0][0][0].shape[-1])
    model = CrisisDetector(embedder, aggregator, detector, embedder_batch_size=16)
    train_test(model, ds, groups, batch_size=1, max_time='00:00:20:00', deterministic=deterministic)

if __name__ == '__main__':
    main()
