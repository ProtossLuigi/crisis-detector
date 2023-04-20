from typing import Any
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as pl
import torchmetrics
from transformers import RobertaForSequenceClassification, AutoConfig

class TextVectorizer(pl.LightningModule):
    def __init__(self, pretrained_name: str, weight: torch.Tensor | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.CrossEntropyLoss(weight)
        self.f1 = torchmetrics.F1Score('binary')
        self.acc = torchmetrics.Accuracy('binary')
        self.prec = torchmetrics.Precision('binary')
        self.rec = torchmetrics.Recall('binary')

        config = AutoConfig.from_pretrained(pretrained_name)
        self.model = RobertaForSequenceClassification.from_pretrained(pretrained_name, num_labels=2, config=config)
    
    def forward(self, x) -> Any:
        return self.model.forward(**x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X).logits
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        acc = self.acc(torch.argmax(y_pred, -1), y)
        f1 = self.f1(torch.argmax(y_pred, -1), y)
        loss = self.loss_fn(y_pred, y)
        self.validation_step_losses.append(loss)
        self.log('val_acc', acc, on_epoch=True)
        self.log('val_f1', f1, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True)
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        acc = self.acc(torch.argmax(y_pred, -1), y)
        score = self.f1(torch.argmax(y_pred, -1), y)
        loss = self.loss_fn(y_pred, y)
        precision = self.prec(torch.argmax(y_pred, -1), y)
        recall = self.rec(torch.argmax(y_pred, -1), y)
        self.log('test_acc', acc, on_epoch=True)
        self.log('test_f1', score, on_epoch=True)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_precision', precision, on_epoch=True)
        self.log('test_recall', recall, on_epoch=True)
    
    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        output = self(batch)
        return output.logits, output.hidden_states[0]
    
    def on_validation_epoch_start(self) -> None:
        self.validation_step_losses = []

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_losses).mean(dim=0)
        self.scheduler.step(loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(optimizer, 'min')
        return optimizer

def main():
    ...

if __name__ == '__main__':
    main()
