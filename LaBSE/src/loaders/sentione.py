from datasets.sentione import SentioneDataset


def _reset_config():
    return dict(
        datadir='../data/sentione/',
        file='shared-Fc702-v-Bq4B-selected-content.json',
    )

_CONFIG = _reset_config()

def load_dataset():
    dataset = SentioneDataset(_CONFIG['datadir'], _CONFIG['file'])
    return dataset
    
    
if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset[1])