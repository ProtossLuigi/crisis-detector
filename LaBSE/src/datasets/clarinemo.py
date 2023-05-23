from datasets.base import TextFileDataset
import csv
import torch


class ClarinEmoDataset(TextFileDataset):
    
    def __init__(self, datadir, filename, data_type='text'):
        self.data_type = data_type # text or sentence
        super().__init__(datadir, filename)
        
    @property
    def label_dict(self):
        return {
            'joy': 0,
            'trust': 1,
            'premonition': 2,
            'surprise': 3,
            'fear': 4,
            'sadness': 5 ,
            'repulsion': 6,
            'anger': 7,
            'positive': 8,
            'negative': 9,
            'neutral': 10, 
        }
        
    def parser(self):
        with open(self.filepath, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            head = next(reader)
            self.X = []
            self.Y = []
            currx, curry = [], []
            for row in reader:
                x = row[0]
                y_sent = [float(n) for n in row[1:len(self.label_dict)+1]]
                y_text = [float(n) for n in row[len(self.label_dict)+2:]]
                if self.data_type == 'text':
                    if '#####' in x:
                        self.X.append(currx[:4])
                        self.Y.append(curry[:4])
                        currx, curry = [], []
                    else:
                        currx.append(x)
                        curry.append(y_sent)
                else:
                    if '#####' in x:
                        continue
                    self.X.append(x)
                    self.Y.append(y_sent)
        return self.X, self.Y

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

    
    