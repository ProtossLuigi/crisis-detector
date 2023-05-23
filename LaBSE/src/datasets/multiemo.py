from email import parser
from datasets.base import TextFileDataset
import re


class MultiEmoDataset(TextFileDataset):
    def __init__(self, datadir, filename):

        values = map(lambda x: x[:-1], re.findall(r'[a-z]+\.', filename))
        order = ['category', 'data_type', 'split', 'language']
        self.model_id = {
            order[i]: v for i, v in enumerate(values)
        }        
        
        super().__init__(datadir, filename)
        
        self.X = [x[0] for x in self.data]
        self.Y = [x[1:] for x in self.data]
        
    @property
    def label_dict(self):
        return {
            'plus_m': 0,
            'minus_m': 1,
            'zero': 2,
            'amb': 3
        }
        
    @property
    def label_mark(self):
        if self.model_id['data_type'] == 'text':
            return '__label__meta_'
        return '__label__z_'

    def parser(self):
        with open(self.filepath, 'r') as f:
            data = f.readlines()
            data = [[s.rstrip() for s in x.split(self.label_mark)]
                    for x in data]
        return data