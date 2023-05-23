from datasets.multiemo import MultiEmoDataset


class PolEmoDataset(MultiEmoDataset):
    def __init__(self, datadir, filename):
        super().__init__(datadir, filename)
        self.label_mark = '__label__z_'