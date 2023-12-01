import torch
from torch import nn

def main():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.fc.out_features = 512
    print(model)
    return model

if __name__ == '__main__':
    main()