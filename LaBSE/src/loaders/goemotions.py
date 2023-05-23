from loaders.utils import create_dataloader
from datasets.goemotions import GoEmotionsDataset
from os.path import join

def _reset_config():
    return dict(
        datadir = '../data/goemotions',
        file_template = '{split}_oh.tsv', #TODO
        languages = ('en',),
    )

_CONFIG = _reset_config()

def _config(**kwargs):
    _CONFIG.update(**kwargs)
    return _CONFIG


def load_train_data():
    config = _config()
    file = join(config['datadir'], config['file_template'].format(split='train'))
    dataset = GoEmotionsDataset(file)
    return create_dataloader(dataset, batch_size=16)


def load_dev_data():
    config = _config()
    file = join(config['datadir'], config['file_template'].format(split='dev'))
    dataset = GoEmotionsDataset(file)
    return create_dataloader(dataset, shuffle=False, batch_size=16)

    
def load_test_data(**kwargs):
    config = _config(**kwargs)
    file = join(config['datadir'], config['file_template'].format(split='test'))
    dataset = GoEmotionsDataset(file)
    return create_dataloader(dataset, shuffle=False)
    

def load_data(**kwargs):
    train = load_train_data()
    devs = load_dev_data()
    test = load_test_data(**kwargs)
    return train, devs, test, _config()['languages']