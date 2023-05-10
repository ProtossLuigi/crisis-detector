from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as pl
import torchmetrics

class AggregatorBackbone(nn.Module):
    def __init__(self, sample_size: int = 0, embedding_dim: int = 768, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
        self.embedding_dim = embedding_dim

class EmbeddingAggregator(pl.LightningModule):
    def __init__(
            self,
            backbone: AggregatorBackbone,
            lr: float = 1e-3,
            weight_decay: float = 0.01,
            class_ratios: torch.Tensor | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = .5 / class_ratios if class_ratios else None

        self.loss_fn = nn.CrossEntropyLoss(self.class_weights)
        self.f1 = torchmetrics.F1Score('multiclass', num_classes=2, average=None)
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=2, average='micro')
        self.prec = torchmetrics.Precision('multiclass', num_classes=2, average=None)
        self.rec = torchmetrics.Recall('multiclass', num_classes=2, average=None)
        
        self.model = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(.1),
            nn.Linear(backbone.embedding_dim, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x) -> Any:
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        X, y = batch
        y_pred = self(X)
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
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        X, y = batch
        y_pred = self(X)
        self.acc.update(torch.argmax(y_pred, -1), y)
        self.f1.update(torch.argmax(y_pred, -1), y)
        self.prec.update(torch.argmax(y_pred, -1), y)
        self.rec.update(torch.argmax(y_pred, -1), y)
        self.log('test_loss', self.loss_fn(y_pred, y), on_epoch=True)
    
    def on_test_epoch_end(self) -> None:
        metrics = {
            'test_acc': self.acc.compute()
        }
        metrics['test_f1_neg'], metrics['test_f1_pos'] = self.f1.compute()
        metrics['test_precision_neg'], metrics['test_precision_pos'] = self.prec.compute()
        metrics['test_recall_neg'], metrics['test_recall_pos'] = self.rec.compute()
        self.log_dict(metrics)
        self.acc.reset()
        self.f1.reset()
        self.prec.reset()
        self.rec.reset()
    
    def configure_optimizers(self) -> Any:
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

class MeanBackbone(AggregatorBackbone):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return torch.mean(x, dim=1)

def main():
    ...

if __name__ == '__main__':
    main()
