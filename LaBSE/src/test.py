from model import LinearClassifier
from loaders import *
import pytorch_lightning as pl
import torch
import h5py
from model_converter import convert


def test(model):
    _,_,test_loader, _ = goemotions.load_data()
    
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        auto_select_gpus=True,
    )
    trainer.test(model=model, dataloaders=test_loader)
    results = model.get_test_results()
    return results


def main():
    ckpt_path = '../models/model.ckpt'
    tf_path = "../models/tf_model.h5"
    
    model = LinearClassifier(no_classes=28)
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    
    results_1 = test(model)
    
    
    model = LinearClassifier(no_classes=28)
    # model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    
    model = convert(tf_path, model)
    
    results_2 = test(model)
    
    for k, v in results_1.items():
        if k in ("macro_f1_score", "macro_accuracy", "macro_recall", "macro_precision"):
            print(k, v)
    print("SECOND--------------------------------------------")
    for k, v in results_2.items():
        if k in ("macro_f1_score", "macro_accuracy", "macro_recall", "macro_precision"):
            print(k, v)
        
        
    
if __name__ == '__main__':
    main()