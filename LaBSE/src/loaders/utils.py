from torch.utils.data import DataLoader
import torch
import random


def create_dataloader(dataset, batch_size=16, shuffle=True,
                      num_workers=8, persistent_workers=True, loader_type=DataLoader, **kwargs):
    return loader_type(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        **kwargs
    )
    

class SequenceLoader():
    
    def __init__(self, dataset, batch_size=16, shuffle=True, *args, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle


    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            random.shuffle(indices)
            
        self.batch_indices = iter([indices[i:i+self.batch_size] 
                                   for i in range(0, len(indices), self.batch_size)])
        return self
    
    def __next__(self):
        indices = next(self.batch_indices)
        max_len = 0
        curr_x = []
        curr_y = []
        for idx in indices:
            x, y = self.dataset[idx]
            curr_x.append(x)
            curr_y.append(y)
            max_len = max(max_len, len(x))
        token = "" if isinstance(curr_x[0][0], str) else 0
        curr_x = [x + [token] * (max_len - len(x)) for x in curr_x]
        padding = [0] * len(curr_y[0][0])
        curr_y = [y + [padding] * (max_len - len(y)) for y in curr_y]
        curr_y = torch.tensor(curr_y)
        return curr_x, curr_y


class MultitaskLoader():
    def __init__(self, datasets, tasks, batch_size=16, shuffle=True, *args, **kwargs):
        self.data = datasets
        self.tasks = tasks
        self.data_no = len(datasets)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        
    def __iter__(self):
        lengths = [len(dataset) for dataset in self.data]
        indices = [list(range(leng)) for leng in lengths]
        if self.shuffle:
            for ind in indices:
                random.shuffle(ind)
        
        all = sum(lengths)
        batch_sizes = torch.tensor([self.batch_size*length//all for length in lengths])
        steps = lengths[0]//batch_sizes[0].item()
        
        self.batch_indices = []
        for i in range(steps):
            curr_iter = torch.tensor([0 for _ in indices])
            curr_batch = [
                indices[i][idx:idx+batch_sizes[i]]
                for i, idx in enumerate(curr_iter.tolist())
            ]
            curr_iter += batch_sizes
            self.batch_indices.append(curr_batch)
            
                
        self.batch_indices = iter(self.batch_indices)
        return self
        
    def __next__(self):
        indices = next(self.batch_indices)
        X, Y = [], []
        for i, dset in enumerate(indices):
            curr_x = []
            curr_y = []
            curr_t = self.tasks[i]
            for idx in dset:
                x, y = self.data[i][idx]
                curr_x.append(x)
                curr_y.append(y)
            X.append((curr_x, curr_t))
            Y.append((torch.stack(curr_y), curr_t))     
        
        return X, Y