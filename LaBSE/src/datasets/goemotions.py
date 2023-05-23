from datasets.base import PandasDataset
import pandas as pd
from torch import tensor

class GoEmotionsDataset(PandasDataset):
    def __init__(self, filepath, ekman=False):
        data = pd.read_csv(filepath, header=0, sep='\t')
        super().__init__(data)
        self.x_column = 'text'
        if ekman:
            self.y_column = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        else:
            self.y_column = self.data.columns[1:]
            
    @property
    def label_dict(self):
        return super().label_dict
    
    def __getitem__(self, index):
        curr = self.data.iloc[index]
        y = [float(o) for o in curr[self.y_column].to_list()]
        return curr[self.x_column], tensor(y)
