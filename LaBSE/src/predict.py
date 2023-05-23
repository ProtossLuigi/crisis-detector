from model import LinearClassifier
import torch
from model_converter import convert

def main():
    ckpt_path = "models/model.ckpt"
    tf_path = "models/tf_model.h5"

    sent = "Ala ma okropnego i obrzydliwego kota"
    model = LinearClassifier(no_classes=28)
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])

    out = model(sent)
    print(out)

    model = convert(tf_path, model)
        
    out = model(sent)
    print(out)
    
    
if __name__ == "__main__":
    main()
