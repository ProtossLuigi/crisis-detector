from typing import Any

import torch
from torch import nn
import lightning as pl

class CrisisDetector(pl.LightningModule):
    def __init__(
            self,
            embedder_model: nn.Module,
            embedder_params: dict,
            aggregator_model: nn.Module,
            aggregator_params: dict,
            detector_model: nn.Module,
            detector_params: dict,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)


def main():
    pass

if __name__ == '__main__':
    main()
