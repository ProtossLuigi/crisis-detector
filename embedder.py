from typing import Any, Tuple
import os
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as pl
from lightning.pytorch import seed_everything
import torchmetrics
from transformers import AutoModel, AutoConfig, AutoTokenizer

from data_tools import load_data, DictDataset, load_text_data, get_verified_data, get_data_with_dates
from training_tools import train_model, test_model, cross_validate, split_dataset

torch.set_float32_matmul_precision('medium')

class TextEmbedder(pl.LightningModule):
    def __init__(self, pretrained_name: str, weight: torch.Tensor | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pretrained_name = pretrained_name

        self.loss_fn = nn.CrossEntropyLoss(weight)
        self.f1 = torchmetrics.F1Score('multiclass', num_classes=2, average='macro')
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=2, average='macro')
        self.prec = torchmetrics.Precision('multiclass', num_classes=2, average='macro')
        self.rec = torchmetrics.Recall('multiclass', num_classes=2, average='macro')

        # config = AutoConfig.from_pretrained(self.pretrained_name)
        # config.num_labels = 2
        # config.output_hidden_states = True
        self.model = AutoModel.from_pretrained(self.pretrained_name)
        self.classifier = nn.Sequential(
            nn.Dropout(.1),
            nn.ReLU(),
            nn.Linear(768, 2)
        )

        self.save_hyperparameters()
    
    def forward(self, *args, **kwargs) -> Any:
        if args:
            x = args[0]
        else:
            x = kwargs
        return self.model(input_ids=x['input_ids'], attention_mask=x['attention_mask'], return_dict=False)
    
    def training_step(self, batch, batch_idx):
        y = batch['label']
        embeddings = self(batch)[0][:, 0]
        y_pred = self.classifier(embeddings)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        y_true = batch['label']
        embeddings = self(batch)[0][:, 0]
        y_pred = self.classifier(embeddings)
        acc = self.acc(torch.argmax(y_pred, -1), y_true)
        f1 = self.f1(torch.argmax(y_pred, -1), y_true)
        loss = self.loss_fn(y_pred, y_true)
        self.validation_step_losses.append(loss)
        self.log('val_acc', acc, on_epoch=True)
        self.log('val_f1', f1, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True)
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        y_true = batch['label']
        embeddings = self(batch)[0][:, 0]
        y_pred = self.classifier(embeddings)
        acc = self.acc(torch.argmax(y_pred, -1), y_true)
        score = self.f1(torch.argmax(y_pred, -1), y_true)
        loss = self.loss_fn(y_pred, y_true)
        precision = self.prec(torch.argmax(y_pred, -1), y_true)
        recall = self.rec(torch.argmax(y_pred, -1), y_true)
        self.log('test_acc', acc, on_epoch=True)
        self.log('test_f1', score, on_epoch=True)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_precision', precision, on_epoch=True)
        self.log('test_recall', recall, on_epoch=True)
    
    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch).hidden_states[0][:, 0, :]
    
    def on_validation_epoch_start(self) -> None:
        self.validation_step_losses = []

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_losses).mean(dim=0)
        self.scheduler.step(loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(optimizer, 'min')
        return optimizer

def create_token_dataset(df: pd.DataFrame, tokenizer_name: str, batch_size: int = 256, max_length: int | None = 256):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    X, y = {}, []
    for i in tqdm(range(0 , df.shape[0], batch_size)):
        X_batch = df['text'][i:i+batch_size]
        y_batch = df['label'][i:i+batch_size]
        X_batch = tokenizer(X_batch.to_list(), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        for key in X_batch:
            if key in X:
                X[key].append(X_batch[key])
            else:
                X[key] = [X_batch[key]]
        y.append(torch.tensor(y_batch.values, dtype=torch.long))
    for key in X:
        X[key] = torch.cat(X[key], dim=0)
    y = torch.cat(y, dim=0)
    ds = DictDataset(X, y)
    tokenizer = None
    return ds

def main():
    TEXTS_PATH = 'saved_objects/texts_df.feather'
    DATASET_PATH = 'saved_objects/token_ds.pt'
    pretrained_name = 'allegro/herbert-base-cased'

    deterministic = True
    end_to_end = False
    batch_size = 128

    if deterministic:
        seed_everything(42)

    if end_to_end or not os.path.isfile(TEXTS_PATH):
        dates = get_data_with_dates(get_verified_data())
        posts_df = load_text_data(dates['path'], dates['crisis_start'], drop_invalid=True)
        posts_df.to_feather(TEXTS_PATH)

        if deterministic:
            seed_everything(42)
    else:
        posts_df = pd.read_feather(TEXTS_PATH)
    
    if end_to_end or not os.path.isfile(DATASET_PATH):
        ds = create_token_dataset(posts_df, pretrained_name)
        torch.save(ds, DATASET_PATH)

        if deterministic:
            seed_everything(42)
    else:
        ds = torch.load(DATASET_PATH)
    
    groups = torch.tensor(posts_df['group'].values)

    train_ds, val_ds, test_ds = split_dataset(ds, groups, stratify=True)
    # class_ratio = train_ds[:]['label'].unique(return_counts=True)[1] / len(train_ds)
    # weight = torch.pow(class_ratio * class_ratio.shape[0], -1)
    weight = None

    model = TextEmbedder(pretrained_name, weight)
    trainer = train_model(model, train_ds, val_ds, batch_size=batch_size, max_epochs=30, deterministic=deterministic)
    # model = TextEmbedder.load_from_checkpoint('checkpoints/epoch=0-step=1828.ckpt', pretrained_name=pretrained_name)
    test_model(train_ds, trainer=trainer, batch_size=batch_size)
    test_model(test_ds, trainer=trainer, batch_size=batch_size)
    
    # cross_validate(TextEmbedder, (pretrained_name,), ds, groups, use_weights=True, n_splits=5, batch_size=64, max_epochs=1, deterministic=deterministic)
    

if __name__ == '__main__':
    main()
