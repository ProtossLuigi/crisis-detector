from typing import Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset, DataLoader
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import GroupKFold

def split_dataset(ds: Dataset, groups: torch.Tensor, n_splits: int = 10, validate: bool = True) -> Tuple[Dataset, Dataset] | Tuple[Dataset, Dataset, Dataset]:
    fold = GroupKFold(n_splits)
    splits = list(fold.split(ds, groups=groups))
    train_idx = splits[0][0]
    test_idx = splits[0][1]
    if validate:
        val_idx = splits[1][1]
        train_idx = np.array(list(set(train_idx) - set(val_idx)))
        return Subset(ds, train_idx), Subset(ds, test_idx), Subset(ds, val_idx)
    else:
        return Subset(ds, train_idx), Subset(ds, test_idx)

def fold_dataset(ds: Dataset, groups: torch.Tensor, n_splits: int = 10, validate: bool = True) -> List[Tuple[Dataset, Dataset]] | List[Tuple[Dataset, Dataset, Dataset]]:
    fold = GroupKFold(n_splits)
    splits = list(fold.split(ds, groups=groups))
    if validate:
        train_idx, test_idx = tuple(zip(*splits))
        val_idx = test_idx[1:] + test_idx[:1]
        train_idx = [np.array(list(set(t) - set(v))) for t, v in zip(train_idx, val_idx)]
        return [(Subset(ds, t), Subset(ds, t2), Subset(ds, v)) for t, v, t2 in zip(train_idx, val_idx, test_idx)]
    else:
        return [(Subset(ds, train_idx), Subset(ds, test_idx)) for train_idx, test_idx in splits]

def train_model(
        model: pl.LightningModule,
        train_ds: Dataset,
        val_ds: Dataset | None = None,
        batch_size: int = 512,
        max_epochs: int = -1,
        max_time = None,
        num_workers: int = 10,
        verbose: bool = True,
        deterministic: bool = False
) -> pl.Trainer:
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_ds else None
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        save_top_k=1
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        logger=False,
        callbacks=[
        checkpoint_callback,
        EarlyStopping(monitor='val_f1', mode='max', patience=20)
        ],
        max_epochs=max_epochs,
        max_time=max_time,
        enable_model_summary=verbose,
        # enable_progress_bar=verbose,
        deterministic=deterministic
    )

    trainer.fit(model, train_dl, val_dl)
    return trainer

def test_model(
        trainer: pl.Trainer,
        test_ds: Dataset,
        batch_size: int = 512,
        num_workers: int = 10,
        checkpoint: bool = True,
        verbose: bool = True
) -> dict:
    ckpt_path = 'best' if checkpoint else None
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return trainer.test(dataloaders=test_dl, ckpt_path=ckpt_path, verbose=verbose)[0]

def cross_validate(
        model_class: type,
        model_params: Tuple,
        ds: Dataset,
        groups: torch.Tensor,
        use_weights: bool = False,
        n_splits: int = 10,
        batch_size: int = 512,
        max_epochs: int = -1,
        max_time = None,
        num_workers: int = 10,
        deterministic: bool = False
) -> pd.DataFrame:
    folds = fold_dataset(ds, groups, n_splits)
    stats = []
    for train_ds, test_ds, val_ds in tqdm(folds):
        if use_weights:
            class_ratio = train_ds[:][1].unique(return_counts=True)[1] / len(train_ds)
            weight = torch.pow(class_ratio * class_ratio.shape[0], -1)
            model_params += (weight,)
        model = model_class(*model_params)

        trainer = train_model(model, train_ds, val_ds, batch_size, max_epochs, max_time, num_workers, False, deterministic)
        stats.append(test_model(trainer, test_ds, batch_size, num_workers, True, False))
    stats = pd.DataFrame(stats)
    print(stats)
    print('Means:')
    print(stats.mean(axis=0))
    print('Standard deviation:')
    print(stats.std(axis=0))
    return stats
