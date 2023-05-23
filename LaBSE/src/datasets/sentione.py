from datasets.base import TextFileDataset
import json

class SentioneDataset(TextFileDataset):
    def __init__(self, datadir, filename):
        super().__init__(datadir, filename)
        self.X = self.data
        
    @property
    def label_dict(self):
         return {}   
        
    def parser(self):
        with open(self.filepath) as f:
            lines = f.readlines()
        data = [json.loads(line) for line in lines]
        return data
        
    def __getitem__(self, idx):
        return self.data[idx]