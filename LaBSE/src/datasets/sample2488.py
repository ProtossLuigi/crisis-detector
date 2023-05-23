from datasets.base import PandasDataset

class Sample2488Dataset(PandasDataset):
    def __init__(self, data):
        super().__init__(data)
        self.x_column = 'text'
        self.y_column = 'sentiment'
    
    @property
    def label_dict(self):
        return {
            k: k-1 for k in range(1, 4)
        }