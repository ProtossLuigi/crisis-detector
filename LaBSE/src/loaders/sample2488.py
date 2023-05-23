from sklearn.model_selection import train_test_split
from datasets.sample2488 import Sample2488Dataset
from loaders.utils import create_dataloader
from sklearn.model_selection import train_test_split
from os.path import join
import pandas as pd


def _reset_config():
    return dict(
        datadir = '../data/sample_2488',
        languages = ('pl',),
        train_test_split=0.0,
        train_dev_split=0.2
    )

_CONFIG = _reset_config()

def _config(**kwargs):
    _CONFIG.update(kwargs)
    return _CONFIG


def load_train_data(train_split):
    return create_dataloader(
        Sample2488Dataset(train_split)
    )
    
def load_dev_data(dev_split):
    return create_dataloader(
        Sample2488Dataset(dev_split),
        shuffle=False
    )
    
    
def load_data(filename):
    config = _config()
    datafile = join(config['datadir'], filename)
    df = pd.read_csv(datafile)
    if config['train_test_split'] > 0:
        train_split, test_split = train_test_split(df, test_size=config['train_test_split'])
        test = load_dev_data(test_split)
    else:
        test = None
    train_split, dev_split = train_test_split(df, test_size=config['train_dev_split'])
    
    train = load_train_data(train_split)
    dev = load_dev_data(dev_split)
    
    return train, dev, test, config['languages']