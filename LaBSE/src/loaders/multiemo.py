from datasets.multiemo import MultiEmoDataset
from loaders.utils import create_dataloader


def _reset_config():
    return dict(
        datadir = '../data/multiemo',
        file_template = '{category}.{data_type}.{split}.{language}.txt',
        languages = ('de', 'en', 'es', 'fr', 'it',
                    'ja', 'nl', 'pl', 'pt-PT', 'ru', 'zh'),
        data_type = 'text',
        category = 'all',
        train_lang = 'pl',
    )

_CONFIG = _reset_config()

def _config(**kwargs):
    _CONFIG.update(kwargs)
    return _CONFIG


def load_train_data():
    config = _config()
    file = config['file_template'].format(
        category=config['category'], data_type=config['data_type'],
        split='train', language=config['train_lang'])
    dataset = MultiEmoDataset(config['datadir'], file)
    return create_dataloader(dataset)


def load_dev_data():
    config = _config()
    files = [config['file_template'].format(
                category=config['category'], data_type=config['data_type'],
                split='dev', language=lang) 
             for lang in config['languages']]
    return [
        create_dataloader(MultiEmoDataset(config['datadir'], f),
                          shuffle=False)
        for f in files
    ]

    
def load_test_data(languages=['pl'], **kwargs):
    config = _config(**kwargs)
    files = [config['file_template'].format(
                category=config['category'], data_type=config['data_type'],
                split='test', language=lang) 
             for lang in languages]
    return [
        create_dataloader(MultiEmoDataset(config['datadir'], f),
                          shuffle=False)
        for f in files
    ]
    

def load_data(**kwargs):
    train = load_train_data()
    devs = load_dev_data()
    test = load_test_data(**kwargs)
    return train, devs, test, _config()['languages']