from numpy import average
import torch
import torchmetrics

class ClassificationMetrics(object):
    def __init__(self, num_classes,
                 metrics=['accuracy', 'recall', 'precision', 'f1_score'],
                 averages=[None, 'micro', 'macro']):
        
        base = {
            'accuracy': torchmetrics.Accuracy,
            'recall': torchmetrics.Recall,
            'precision': torchmetrics.Precision,
            'f1_score': torchmetrics.F1Score,
        }
        self.__metric_types = {k: base[k] for k in metrics}
        self.num_classes = num_classes
        self.metrics = {
            'per_class' if k is None else k: self._init_metrics(k) for k in averages
        }        
    
    def _init_metrics(self, average):
        return {
            k: v(task='multilabel', num_labels=self.num_classes, average=average) 
            for k, v in self.__metric_types.items()
        }
        
    def reset_metrics(self):
        for k, v in self.metrics.items():
            for k1, v1 in v.items():
                v1.reset()
    
    def update_metrics(self, preds, labels):
        labels = self.flatten_tensor(labels.type(torch.int16))
        for k, v in self.metrics.items():
            for k1, v1 in v.items():
                v1.update(self.flatten_tensor(preds), labels)

    def flatten_metrics(self, computed):
        flattened = {
            f'{name}_{type_}': value for name, metrics in computed.items()
            for type_, value in metrics.items()
        }
        return flattened
            
    def compute_one_metric(self, metric):
        result = metric.compute()
        if len(result.shape) == 0:
            return result.item()
        return [i.item() for i in list(result)]

    def compute_metrics(self, reset=True, flatten=False):
        metric_values = {
            k: {k1: self.compute_one_metric(v1) for k1, v1 in v.items()} 
            for k, v in self.metrics.items()
        }
        if reset:
            self.reset_metrics()
        if flatten:
            return self.flatten_metrics(metric_values)
        return metric_values
    
    def flatten_tensor(self, tensor_):
        return tensor_.view(-1, tensor_.shape[-1])