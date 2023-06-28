import os
from typing import Any, Tuple
import pandas as pd
from tqdm import tqdm
from math import ceil

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import lightning as pl
import torchmetrics
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from embedder import TextEmbedder
from crisis_detector import Shift2Metric, ShiftMetric
from data_tools import SeriesDataset, get_data_with_dates, get_verified_data, load_text_data
from training_tools import init_trainer, split_dataset
from periodic_activations import SineActivation

torch.set_float32_matmul_precision('high')

class PostDetector(pl.LightningModule):
    def __init__(
            self, 
            time_model: nn.Module, 
            embedder_model: TextEmbedder, 
            classifier_model: nn.Module, 
            train_dataloader_len: int | None = None,
            warmup_proportion: float = .1,
            max_epochs: int | None = None,
            class_ratios: torch.Tensor | None = None,
            *args: Any, 
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.train_dataloader_len = train_dataloader_len
        self.warmup_proportion = warmup_proportion
        self.max_epochs = max_epochs

        self.time_vectorizer = time_model
        self.embedder = embedder_model
        self.classifier = classifier_model

        self.f1 = torchmetrics.F1Score('multiclass', num_classes=2, average=None)
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=2, average='micro')
        self.prec = torchmetrics.Precision('multiclass', num_classes=2, average=None)
        self.rec = torchmetrics.Recall('multiclass', num_classes=2, average=None)
        self.shift = ShiftMetric()
        self.shift2 = Shift2Metric()

        self.init_loss_fn(class_ratios)

        self.save_hyperparameters()
    
    def init_loss_fn(self, class_ratios: torch.Tensor | None = None) -> None:
        self.class_weights = .5 / class_ratios if class_ratios is not None else None
        self.loss_fn = nn.CrossEntropyLoss(self.class_weights)
    
    def forward(self, timestamps, input_ids, attention_mask, features = None) -> Any:
        tvec = self.time_vectorizer(timestamps)
        embeddings = self.embedder(input_ids=input_ids.flatten(0, 1), attention_mask=attention_mask.flatten(0, 1)).view(tvec.shape[0], tvec.shape[1], -1)
        if features is None:
            return self.classifier(torch.cat((tvec, embeddings), dim=-1))
        else:
            return self.classifier(torch.cat((tvec, embeddings, features), dim=-1))
    
    def training_step(self, batch, batch_idx):
        if len(batch) == 5:
            timestamps, input_ids, attention_mask, features, y = batch
        else:
            timestamps, input_ids, attention_mask, y = batch
            features = None
        y_pred = self(timestamps, input_ids, attention_mask, features)
        loss = self.loss_fn(y_pred, y)
        self.acc.update(torch.argmax(y_pred, -1), y)
        self.f1.update(torch.argmax(y_pred, -1), y)
        f1 = self.f1.compute()
        metrics = {
            'train_loss': loss.detach(),
            'train_acc': self.acc.compute(),
            'train_f1_0': f1[0],
            'train_f1_1': f1[1],
            'train_f1_macro': f1.mean(),
        }
        self.log_dict(metrics, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.acc.reset()
        self.f1.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if len(batch) == 5:
            timestamps, input_ids, attention_mask, features, y = batch
        else:
            timestamps, input_ids, attention_mask, y = batch
            features = None
        y_pred = self(timestamps, input_ids, attention_mask, features)
        self.acc.update(torch.argmax(y_pred, -1), y)
        self.f1.update(torch.argmax(y_pred, -1), y)
        self.log('val_loss', self.loss_fn(y_pred, y), on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        metrics = {
            'val_acc': self.acc.compute(),
            'val_f1_macro': self.f1.compute().mean()
        }
        self.log_dict(metrics)
        self.acc.reset()
        self.f1.reset()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if len(batch) == 5:
            timestamps, input_ids, attention_mask, features, y = batch
        else:
            timestamps, input_ids, attention_mask, y = batch
            features = None
        y_pred = self(timestamps, input_ids, attention_mask, features)
        # if self.print_test_samples:
        #     print(y)
        #     print(torch.argmax(y_pred, -1))
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
        f1 = self.f1.compute()
        metrics['test_f1_0'], metrics['test_f1_1'], metrics['test_f1_macro'] = f1[0], f1[1], f1.mean(dim=0)
        prec = self.prec.compute()
        metrics['test_precision_0'], metrics['test_precision_1'] = prec[0], prec[1]
        rec = self.rec.compute()
        metrics['test_recall_0'], metrics['test_recall_1'] = rec[0], rec[1]
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
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return torch.argmax(self(batch[0]), dim=-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            # {'params': self.embedder.parameters(), 'lr': 1e-5},
            {'params': self.time_vectorizer.parameters(), 'lr': 1e-3},
            {'params': self.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=.01)
        if self.warmup_proportion is not None and self.train_dataloader_len is not None:
            num_training_steps = self.max_epochs * self.train_dataloader_len
            num_warmup_steps = int(self.warmup_proportion * num_training_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer

class LSTMBackbone(nn.Module):
    def __init__(self, input_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Dropout(.1),
            nn.PReLU()
        )
        self.lstm = nn.LSTM(512, 512, num_layers=1, batch_first=True)
        self.output_net = nn.Sequential(
            nn.Dropout(.1),
            nn.PReLU(),
            nn.Linear(512, 128),
            nn.Dropout(.1),
            nn.PReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.output_net(self.lstm(self.input_net(x))[0][:, -1])

def create_dataset(
        df: pd.DataFrame, 
        tokenizer_name: str, 
        sequence_len: int = 50, 
        text_length: int = 256, 
        batch_size: int = 128, 
        window_size: int | Tuple[int, int] | None = None, 
        sequence_step: int = 1,
        balance_classes: bool = False
):
    if window_size is None:
        window_size = (sequence_len * 2 - 1, sequence_len)
    elif type(window_size) == int:
        window_size = (window_size + sequence_len - 1, window_size)

    times = torch.tensor(df['time'].apply(lambda x: x.timetuple()[:6]), dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    texts = df['text']
    input_ids = []
    attention_mask = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Tokenizing'):
        batch = texts[i:i+batch_size].to_list()
        tokens = tokenizer(batch, truncation=True, padding='max_length', max_length=text_length, return_tensors='pt')
        input_ids.append(tokens['input_ids'])
        attention_mask.append(tokens['attention_mask'])
    input_ids = torch.cat(input_ids)
    attention_mask = torch.cat(attention_mask)

    sentiments = ['brak', 'negatywny', 'neutralny', 'pozytywny']
    sentiment_data = pd.get_dummies(df['sentiment'])
    for sent in sentiments:
        if sent not in sentiment_data.columns:
            sentiment_data[sent] = False
    sentiment_data = torch.tensor(sentiment_data.values, dtype=torch.float32)
    features = sentiment_data

    labels = torch.tensor(df['time_label'], dtype=torch.long)
    groups = torch.tensor(df['group'].values)

    times_seq = []
    input_ids_seq = []
    attention_mask_seq = []
    features_seq = []
    labels_seq = []
    groups_seq = []

    for g in tqdm(groups.unique(), desc='Generating sequences'):
        selector0 = ((groups == g) & ~labels).nonzero().flatten()[-window_size[0]:]
        selector1 = ((groups == g) & labels).nonzero().flatten()[:window_size[1]]
        if balance_classes:
            selector_len = min(len(selector0) - sequence_len + 1, len(selector1))
            if ceil(2 * selector_len / sequence_step) % 2 == 1:
                target_seq_num = ceil(2 * selector_len / sequence_step) - 1
                selector_len = target_seq_num * sequence_step // 2
            selector0 = selector0[-(2 * selector_len - 1):]
            selector1 = selector1[:selector_len]
        selector = torch.zeros_like(labels, dtype=bool)
        selector[torch.cat((selector0, selector1))] = True
        times_g = times[selector]
        input_ids_g = input_ids[selector]
        attention_mask_g = attention_mask[selector]
        features_g = features[selector]
        labels_g = labels[selector]
        for i in range(0, labels_g.shape[0] - sequence_len + 1, sequence_step):
            times_seq.append(times_g[i:i+sequence_len])
            input_ids_seq.append(input_ids_g[i:i+sequence_len])
            attention_mask_seq.append(attention_mask_g[i:i+sequence_len])
            features_seq.append(features_g[i:i+sequence_len])
            labels_seq.append(labels_g[i+sequence_len-1])
            groups_seq.append(g)
    
    return TensorDataset(torch.stack(times_seq), torch.stack(input_ids_seq), torch.stack(attention_mask_seq), torch.stack(features_seq), torch.stack(labels_seq)), torch.tensor(groups_seq)

def train_test(
        model: PostDetector, 
        ds: Dataset, 
        groups: torch.Tensor, 
        batch_size: int = 1, 
        precision: str = 'bf16-mixed', 
        max_epochs: int = -1,
        max_time: Any | None = None, 
        deterministic: bool = False
):
    trainer = init_trainer(
        precision,
        early_stopping=True, 
        logging={'name': 'crisis-detector-v2', 'project': 'crisis-detector'}, 
        max_epochs=max_epochs, 
        max_time=max_time, 
        accumulate_grad_batches=16,
        deterministic=deterministic
    )
    train_ds, test_ds, val_ds = split_dataset(ds, groups, n_splits=10, validate=True)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=10, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=10, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=1, pin_memory=True)
    class_ratios = train_ds[:][-1].unique(return_counts=True)[-1] / len(train_ds)
    print(class_ratios)
    model.init_loss_fn(class_ratios)
    model.train_dataloader_len = len(train_dl)
    model.max_epochs = max_epochs
    trainer.fit(model, train_dl, val_dl) 
    trainer.test(model, test_dl, 'best')

def main():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    deterministic = True
    end_to_end = True
    samples_limit = None
    batch_size = 1
    max_epochs = 100
    
    TEXTS_PATH = 'saved_objects/texts_df' + str(samples_limit) + '.feather'
    # DATASET_PATH = 'saved_objects/token_ds' + str(samples_limit) + '.pt'
    
    embedder = TextEmbedder.load_from_checkpoint('saved_objects/pretrained_herbert.ckpt')
    time_vectorizer = SineActivation(6, 64)
    classifier = LSTMBackbone(836)

    if deterministic:
        pl.seed_everything(42)

    if end_to_end or not os.path.isfile(TEXTS_PATH):
        dates = get_data_with_dates(get_verified_data())
        posts_df = load_text_data(dates['path'], dates['crisis_start'], samples_limit=samples_limit, drop_invalid=True)
        posts_df.to_feather(TEXTS_PATH)

        if deterministic:
            pl.seed_everything(42)
    else:
        posts_df = pd.read_feather(TEXTS_PATH)
    
    ds, groups = create_dataset(posts_df, embedder.pretrained_name, sequence_len=100, sequence_step=5, balance_classes=True)

    model = PostDetector(time_vectorizer, embedder, classifier)
    train_test(model, ds, groups, batch_size, '16-mixed', max_epochs=max_epochs, deterministic=deterministic)

if __name__ == '__main__':
    main()
