import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch import seed_everything
from transformers import AutoTokenizer

import embedder
import aggregator
import crisis_detector
from data_tools import SeriesDataset, get_data_with_dates, get_verified_data, get_all_data, load_text_data, load_data

def triple_training():
    deterministic = True
    end_to_end = False
    samples_limit = 1000
    embedder_training_batch_size = 128
    aggregator_training_batch_size = 512
    day_post_sample_size = 50
    padding = False
    
    TEXTS1_PATH = 'saved_objects/texts_df' + str(samples_limit) + '.feather'
    DATASET_PATH = 'saved_objects/token_ds' + str(samples_limit) + '.pt'
    pretrained_name = 'allegro/herbert-base-cased'

    TEXTS2_PATH = 'saved_objects/texts_no_sample_df.feather'
    EMBEDDINGS_PATH = 'saved_objects/embeddings_all.pt'

    DAYS_DF_PATH = 'saved_objects/days_df.feather'
    POSTS_DF_PATH = 'saved_objects/posts_df' + str(day_post_sample_size) + '.feather'
    EMBEDDINGS_PATH = 'saved_objects/embeddings' + str(day_post_sample_size) + '.pt'

    if deterministic:
        seed_everything(42)

    if end_to_end or not os.path.isfile(TEXTS1_PATH):
        dates = get_data_with_dates(get_verified_data())
        posts_df = load_text_data(dates['path'], dates['crisis_start'], samples_limit=samples_limit, drop_invalid=True)
        posts_df.to_feather(TEXTS1_PATH)

        if deterministic:
            seed_everything(42)
    else:
        posts_df = pd.read_feather(TEXTS1_PATH)
    
    if end_to_end or not os.path.isfile(DATASET_PATH):
        ds = embedder.create_token_dataset(posts_df, pretrained_name)
        torch.save(ds, DATASET_PATH)

        if deterministic:
            seed_everything(42)
    else:
        ds = torch.load(DATASET_PATH)
    
    groups = torch.tensor(posts_df['group'].values)

    embedder_model = embedder.TextEmbedder(pretrained_name)
    tokenizer = AutoTokenizer.from_pretrained(embedder_model.pretrained_name)
    embedder.train_test(embedder_model, ds, groups, embedder_training_batch_size, max_epochs=150, deterministic=deterministic)

    if deterministic:
        seed_everything(42)

    if end_to_end or not os.path.isfile(TEXTS2_PATH):
        dates = get_data_with_dates(get_verified_data())
        posts_df = aggregator.load_data(dates['path'], dates['crisis_start'], drop_invalid=True)
        posts_df.to_feather(TEXTS2_PATH)

        if deterministic:
            seed_everything(42)
    else:
        posts_df = pd.read_feather(TEXTS2_PATH)
    
    if end_to_end or not os.path.isfile(EMBEDDINGS_PATH):
        ds = SeriesDataset(posts_df['text'])
        collate_fn = lambda x: tokenizer(x, truncation=True, padding=True, max_length=256, return_tensors='pt')
        dl = DataLoader(ds, 128, num_workers=10, collate_fn=collate_fn, pin_memory=True)
        trainer = pl.Trainer(devices=1, precision='bf16-mixed', logger=False, deterministic=deterministic)
        embeddings = trainer.predict(embedder_model, dl)
        embeddings = torch.cat(embeddings, dim=0)

        with open(EMBEDDINGS_PATH, 'wb') as f:
            torch.save(embeddings, f)
        
        if deterministic:
            seed_everything(42, workers=True)
    else:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = torch.load(f)
    
    ds, groups = aggregator.create_dataset(posts_df, embeddings, .02, day_post_sample_size, aggregator_training_batch_size > 0, padding, balance_classes=True)
    aggregator_model = aggregator.TransformerAggregator(sample_size=day_post_sample_size)
    aggregator.train_test(aggregator_model, ds, groups, batch_size=aggregator_training_batch_size, max_epochs=150, deterministic=deterministic)

    posts_df = None
    ds = None
    dl = None
    embeddings = None

    if deterministic:
        seed_everything(42, workers=True)

    if end_to_end or not (os.path.isfile(DAYS_DF_PATH) and os.path.isfile(POSTS_DF_PATH)):

        data = get_data_with_dates(get_all_data())

        days_df, text_df = load_data(data, day_post_sample_size, True, None, (59, 30))
        days_df.to_feather(DAYS_DF_PATH)
        text_df.to_feather(POSTS_DF_PATH)

        if deterministic:
            seed_everything(42, workers=True)
    else:
        days_df = pd.read_feather(DAYS_DF_PATH)
        text_df = pd.read_feather(POSTS_DF_PATH)

    if end_to_end or not os.path.isfile(EMBEDDINGS_PATH):
        ds = SeriesDataset(text_df['text'])
        collate_fn = lambda x: tokenizer(x, truncation=True, padding=True, max_length=256, return_tensors='pt')
        dl = DataLoader(ds, 64, num_workers=10, collate_fn=collate_fn, pin_memory=True)
        trainer = pl.Trainer(devices=1, precision='bf16-mixed', logger=False, deterministic=deterministic)
        embeddings = trainer.predict(embedder_model, dl, ckpt_path='best')
        embeddings = torch.cat(embeddings, dim=0)

        with open(EMBEDDINGS_PATH, 'wb') as f:
            torch.save(embeddings, f)
        
        if deterministic:
            seed_everything(42, workers=True)
    else:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = torch.load(f)

    days_df = crisis_detector.add_embeddings(days_df, text_df, embeddings, aggregator_model, 1024, deterministic=deterministic)
    ds, groups = crisis_detector.create_dataset(days_df, 30)

    if deterministic:
        seed_everything(42, workers=True)
    
    detector_model = crisis_detector.MyTransformer(print_test_samples=True)
    crisis_detector.train_test(detector_model, ds, groups, batch_size=512, max_epochs=150, deterministic=deterministic)

if __name__ == '__main__':
    triple_training()
