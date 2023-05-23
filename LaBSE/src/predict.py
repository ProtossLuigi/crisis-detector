from model import LinearClassifier
import torch
from model_converter import convert
from tqdm import tqdm

def main():
    data_path = '/home/proto/repos/crisis-detector/saved_objects/texts_full.txt'
    tf_path = "LaBSE/models/tf_model.h5"
    batch_size = 4

    with open(data_path, 'r') as f:
        texts = f.readlines()
    model = LinearClassifier(no_classes=28)
    # model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])

    # out = model(sent)
    # print(out)

    model = convert(tf_path, model)
    outs = []

    model.eval()
    torch.no_grad()

    for i in tqdm(range(0, len(texts), batch_size)):
        out = model(texts[i:i+batch_size])
        outs.append(out[0].cpu())
    torch.save(torch.cat(outs, dim=0), '/home/proto/repos/crisis-detector/saved_objects/emotions.pt')
    
    
if __name__ == "__main__":
    main()
