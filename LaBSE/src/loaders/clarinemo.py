from loaders.utils import create_dataloader, SequenceLoader
from datasets.clarinemo import ClarinEmoDataset
from sklearn.model_selection import train_test_split


def _reset_config():
    return dict(
        datadir = '../data/clarinemo',
        filepath = 'clarinemo_voting.csv',
        languages = ('pl',),
        train_test_split = 0.2,
        train_dev_split = 0.2,
    )

_CONFIG = _reset_config()

def _config(**kwargs):
    _CONFIG.update(kwargs)
    return _CONFIG


def load_data(data_type, **kwargs):
    config = _config()
    dataset = ClarinEmoDataset(config['datadir'], config['filepath'], data_type)
    train_set, test_set = train_test_split(dataset, test_size=config['train_test_split'])
    train_set, dev_set = train_test_split(train_set, test_size=config['train_dev_split'])
    return create_dataloader(train_set, loader_type=SequenceLoader, **kwargs),\
           create_dataloader(dev_set, loader_type=SequenceLoader, **kwargs), \
           create_dataloader(test_set, loader_type=SequenceLoader, **kwargs), \
           config['languages']
    

