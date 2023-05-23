import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
from measurements.metrics import ClassificationMetrics

import pytorch_lightning as pl

class LinearClassifier(pl.LightningModule):
    """
    Model
    """

    def __init__(
        self,
        no_classes: int,
        cuda: bool = True,
        transformer_name: str = "sentence-transformers/LaBSE",
        max_length: int = 512,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.no_classes = no_classes
        self.is_cuda = torch.cuda.is_available() and cuda

        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.transformer = AutoModel.from_pretrained(transformer_name)
        self.embedding_dimension = self.transformer.config.hidden_size
        self.classifier = nn.Linear(self.embedding_dimension, self.no_classes)
        self.activation = torch.nn.Sigmoid()

        if self.is_cuda:
            self.transformer = self.transformer.to("cuda")
            self.classifier = self.classifier.to("cuda")

        self.max_length = max_length
        self.treshold = threshold
        
        
        self.test_metrics = ClassificationMetrics(self.no_classes)

    def _prepare_input(self, texts):
        """
        Function to prepare input texts to tokenizing. It changes collection of texts to list and raw string into list with one element.
        """
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    def _tokenize(self, input_):
        tokens = self.tokenizer.batch_encode_plus(
            input_,
            padding="longest",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.is_cuda:
            tokens = tokens.to("cuda")
        return tokens

    def _output_integers(self, out):
        """ """
        preds = torch.zeros_like(out)
        mask = out >= self.treshold
        preds[mask] = 1
        return preds

    def forward(self, texts):
        input_ = self._prepare_input(texts)
        tokens = self._tokenize(input_)
        embeddings = self.transformer(**tokens).pooler_output
        out = self.classifier(embeddings)
        out = self.activation(out)
        preds = self._output_integers(out)
        return preds, out

    def _measurement_test_metrics(self, preds=None, labels=None,
                                  reset=True):
        if preds is not None and labels is not None:
            preds = preds.to('cpu')
            labels = labels.to('cpu')
            self.test_metrics.update_metrics(preds, labels)
        computed = self.test_metrics.compute_metrics(reset=reset, flatten=True)
        
        return computed


    def test_step(self, batch, batch_idx):
        texts, labels = batch
        preds, _ = self(texts)
        self._measurement_test_metrics(preds, labels, reset=False)
        
    def get_test_results(self):
        return self._measurement_test_metrics(reset=True)