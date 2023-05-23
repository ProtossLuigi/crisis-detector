from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import torch
from os.path import join


class BaseEmotionDataset(ABC, Dataset):
    def __init__(self, data) -> None:
        self.data = data
    
    def _process_labels_to_onehot(self, labels):
        onehot = torch.zeros(len(self.label_dict), dtype=torch.float32)
        for lab in labels:
            onehot[self.label_dict[lab]] = 1.
        return onehot
    
    @property
    @abstractmethod
    def label_dict(self):
        pass
    
    
class PandasDataset(BaseEmotionDataset, ABC):
   
    def __init__(self, data):
        super().__init__(data)
        self.x_column = None
        self.y_column = None
        
    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index) :
        curr = self.data.iloc[index]
        y = self._process_labels_to_onehot([curr[self.y_column].item()])
        return curr[self.x_column], y


class TextFileDataset(BaseEmotionDataset, ABC):
    
    def __init__(self, datadir, filename):        
        self.filepath = join(datadir, filename)
        self.data = self.parser()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        curr_label = self.Y[index]
        y = self._process_labels_to_onehot(curr_label)
        return x, y
    
    @abstractmethod
    def parser(self):
        pass