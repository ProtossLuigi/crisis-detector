from datasets.multiemo import MultiEmoDataset
from loaders.utils import create_dataloader


def _reset_config():
    return dict(
        datadir = '../data/polemo/dataset_conll',
        file_template = '{category}.{data_type}.{split}.txt',
        languages = ('pl',),
        data_type = 'sentence',
        category = 'all',
    )

_CONFIG = _reset_config()

def _config(**kwargs):
    _CONFIG.update(kwargs)
    return _CONFIG


def load_train_data(config=None):
    config = _config() if config is None else config
    file = config['file_template'].format(
        category=config['category'], data_type=config['data_type'],
        split='train')
    dataset = MultiEmoDataset(config['datadir'], file)
    return create_dataloader(dataset)


def load_dev_data():
    config = _config()
    file = config['file_template'].format(
                category=config['category'], data_type=config['data_type'],
                split='dev')
    return [create_dataloader(MultiEmoDataset(config['datadir'], file),
            shuffle=False)]

    
def load_test_data(**kwargs):
    config = _config(**kwargs)
    file = config['file_template'].format(
        category=config['category'], data_type=config['data_type'],
        split='test')
    dataset = MultiEmoDataset(config['datadir'], file)
    return create_dataloader(dataset, shuffle=False)
    

def load_data(test_config=None):
    train = load_train_data()
    devs = load_dev_data()
    if test_config is None:
        test_config = {}
    test = load_test_data(**test_config)
    return train, devs, test, _config()['languages']