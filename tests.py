import os
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch import seed_everything
from transformers import AutoTokenizer

import embedder
import aggregator
import crisis_detector
from data_tools import SeriesDataset, get_data_with_dates, get_verified_data, load_text_data, load_data, FOLD_FILE

def triple_training():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    deterministic = True
    samples_limit = 1000
    embedder_training_batch_size = 128
    aggregator_training_batch_size = 512
    day_post_sample_size = 50
    padding = False
    pretrained_name = 'allegro/herbert-base-cased'

    if deterministic:
        seed_everything(42)

    dates = get_data_with_dates(get_verified_data(2))
    posts_df = load_text_data(dates['path'], dates['crisis_start'], samples_limit=samples_limit, drop_invalid=True)

    if deterministic:
        seed_everything(42)
    
    ds = embedder.create_token_dataset(posts_df, pretrained_name)

    if deterministic:
        seed_everything(42)
    
    groups = torch.tensor(posts_df['group'].values)

    embedder_model = embedder.TextEmbedder(pretrained_name)
    tokenizer = AutoTokenizer.from_pretrained(embedder_model.pretrained_name)
    embedder.train_test(embedder_model, ds, groups, embedder_training_batch_size, max_epochs=150, deterministic=deterministic, predefined=True)

    if deterministic:
        seed_everything(42)

    posts_df = load_text_data(dates['path'], dates['crisis_start'], drop_invalid=True)

    if deterministic:
        seed_everything(42)
    
    ds = SeriesDataset(posts_df['text'])
    collate_fn = lambda x: tokenizer(x, truncation=True, padding=True, max_length=256, return_tensors='pt')
    dl = DataLoader(ds, 128, num_workers=10, collate_fn=collate_fn, pin_memory=True)
    trainer = pl.Trainer(devices=1, precision='bf16-mixed', logger=False, deterministic=deterministic)
    embeddings = trainer.predict(embedder_model, dl)
    embeddings = torch.cat(embeddings, dim=0)
        
    if deterministic:
        seed_everything(42, workers=True)
    
    ds, groups = aggregator.create_dataset(posts_df, embeddings, .02, day_post_sample_size, aggregator_training_batch_size > 0, padding, balance_classes=True)
    aggregator_model = aggregator.TransformerAggregator(sample_size=day_post_sample_size)
    aggregator.train_test(aggregator_model, ds, groups, batch_size=aggregator_training_batch_size, max_epochs=150, deterministic=deterministic, predefined=True)

    posts_df = None
    ds = None
    dl = None
    embeddings = None

    if deterministic:
        seed_everything(42, workers=True)

    days_df, text_df = load_data(dates, day_post_sample_size, True, None, (59, 30))

    if deterministic:
        seed_everything(42, workers=True)

    ds = SeriesDataset(text_df['text'])
    collate_fn = lambda x: tokenizer(x, truncation=True, padding=True, max_length=256, return_tensors='pt')
    dl = DataLoader(ds, 64, num_workers=10, collate_fn=collate_fn, pin_memory=True)
    trainer = pl.Trainer(devices=1, precision='bf16-mixed', logger=False, deterministic=deterministic)
    embeddings = trainer.predict(embedder_model, dl)
    embeddings = torch.cat(embeddings, dim=0)
        
    if deterministic:
        seed_everything(42, workers=True)

    days_df = crisis_detector.add_embeddings(days_df, text_df, embeddings, aggregator_model, 1024, deterministic=deterministic)
    ds, groups = crisis_detector.create_dataset(days_df, 30)

    if deterministic:
        seed_everything(42, workers=True)
    
    detector_model = crisis_detector.MyTransformer(print_test_samples=True)
    crisis_detector.train_test(detector_model, ds, groups, batch_size=512, max_epochs=150, deterministic=deterministic, predefined=True)

def full_cross_validation():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    deterministic = True
    samples_limit = 1000
    embedder_training_batch_size = 128
    aggregator_training_batch_size = 512
    day_post_sample_size = 50
    padding = False
    pretrained_name = 'allegro/herbert-base-cased'

    folds = pd.read_feather(FOLD_FILE)
    results = []

    for i in tqdm(range(len(folds)), desc='Split'):
        split = folds.iloc[i:i+1].reset_index(drop=True)

        if deterministic:
            seed_everything(42)

        dates = get_data_with_dates(get_verified_data(2))
        posts_df = load_text_data(dates['path'], dates['crisis_start'], samples_limit=samples_limit, drop_invalid=True)

        if deterministic:
            seed_everything(42)
        
        ds = embedder.create_token_dataset(posts_df, pretrained_name)

        if deterministic:
            seed_everything(42)
        
        groups = torch.tensor(posts_df['group'].values)

        embedder_model = embedder.TextEmbedder(pretrained_name)
        tokenizer = AutoTokenizer.from_pretrained(embedder_model.pretrained_name)
        embedder.train_test(embedder_model, ds, groups, embedder_training_batch_size, max_epochs=150, deterministic=deterministic, predefined=split)

        if deterministic:
            seed_everything(42)

        posts_df = aggregator.load_data(dates['path'], dates['crisis_start'], drop_invalid=True)

        if deterministic:
            seed_everything(42)
        
        ds = SeriesDataset(posts_df['text'])
        collate_fn = lambda x: tokenizer(x, truncation=True, padding=True, max_length=256, return_tensors='pt')
        dl = DataLoader(ds, 128, num_workers=10, collate_fn=collate_fn, pin_memory=True)
        trainer = pl.Trainer(devices=1, precision='bf16-mixed', logger=False, deterministic=deterministic)
        embeddings = trainer.predict(embedder_model, dl)
        embeddings = torch.cat(embeddings, dim=0)
            
        if deterministic:
            seed_everything(42, workers=True)
        
        ds, groups = aggregator.create_dataset(posts_df, embeddings, .02, day_post_sample_size, aggregator_training_batch_size > 0, padding, balance_classes=True)
        aggregator_model = aggregator.TransformerAggregator(sample_size=day_post_sample_size)
        aggregator.train_test(aggregator_model, ds, groups, batch_size=aggregator_training_batch_size, max_epochs=150, deterministic=deterministic, predefined=split)

        posts_df = None
        ds = None
        dl = None
        embeddings = None

        if deterministic:
            seed_everything(42, workers=True)

        days_df, text_df = load_data(dates, day_post_sample_size, True, None, (59, 30))

        if deterministic:
            seed_everything(42, workers=True)

        ds = SeriesDataset(text_df['text'])
        collate_fn = lambda x: tokenizer(x, truncation=True, padding=True, max_length=256, return_tensors='pt')
        dl = DataLoader(ds, 64, num_workers=10, collate_fn=collate_fn, pin_memory=True)
        trainer = pl.Trainer(devices=1, precision='bf16-mixed', logger=False, deterministic=deterministic)
        embeddings = trainer.predict(embedder_model, dl)
        embeddings = torch.cat(embeddings, dim=0)
            
        if deterministic:
            seed_everything(42, workers=True)

        days_df = crisis_detector.add_embeddings(days_df, text_df, embeddings, aggregator_model, 1024, deterministic=deterministic)
        ds, groups = crisis_detector.create_dataset(days_df, 30)

        if deterministic:
            seed_everything(42, workers=True)
        
        detector_model = crisis_detector.MyTransformer(print_test_samples=True)
        result = crisis_detector.train_test(detector_model, ds, groups, batch_size=512, max_epochs=150, deterministic=deterministic, predefined=split)
        results.append(result)
    
    results = pd.DataFrame(results)
    print(results)
    print('Means:')
    print(results.mean(axis=0))
    print('Standard deviation:')
    print(results.std(axis=0))

if __name__ == '__main__':
    triple_training()
