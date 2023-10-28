import torch
import legacy
import dnnlib
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, in_channels, output_size):
        super(Encoder, self).__init__()
        # Nb of filters for each layer
        self.oneto2 = 32
        self.twoto3 = 64
        self.threeto4 = 128
        self.fourto5 = 128
        self.fiveto6 = 64
        self.sixto7 = 32
        # FC Layers
        self.convtoFC = 32 * 16 * 128
        self.sevento8 = 1024
        self.eightto9 = 512
        # Non-linearity at the end only for the instruments
        self.mysoftmax = nn.Softmax(dim=1)

        # Convolutional layers
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, self.oneto2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(self.oneto2),
            nn.Conv2d(self.oneto2, self.twoto3, kernel_size=3, stride=(2, 1), padding=1),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(self.twoto3),
            nn.Conv2d(self.twoto3, self.threeto4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.BatchNorm2d(self.threeto4),
            nn.Conv2d(self.threeto4, self.fourto5, kernel_size=3, stride=(2, 1), padding=1),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.BatchNorm2d(self.fourto5),
            nn.Conv2d(self.fourto5, self.fiveto6, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer6 = nn.Sequential(
            nn.BatchNorm2d(self.fiveto6),
            nn.Conv2d(self.fiveto6, self.sixto7, kernel_size=3, stride=(2, 1), padding=1),
            nn.LeakyReLU())

        # Linear layers
        self.layer7 = nn.Sequential(
            nn.BatchNorm1d(self.convtoFC),
            nn.Linear(self.convtoFC, self.sevento8),
            nn.LeakyReLU())
        self.layer8 = nn.Sequential(
            nn.BatchNorm1d(self.sevento8),
            nn.Linear(self.sevento8, self.eightto9),
            nn.LeakyReLU())
        self.layer9 = nn.Linear(self.eightto9, output_size)

        # Group layers by kind
        self.convlayers = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6)
        self.fclayers = nn.Sequential(self.layer7, self.layer8, self.layer9)

    def forward(self, x):
        output = self.convlayers(x)
        output = output.reshape(output.size(0), -1)
        output = self.fclayers(output)
        return output

if __name__ == '__main__':
    input_size = 512
    output_size = 1024
    batch_size = 4

    epochs = 10
    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl"

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    label = torch.zeros([1, G.c_dim], device=device)

    for epoch in range(epochs):
        print('Epoch', epoch)
        z = torch.randn(batch_size, input_size).cuda()
        img = G(z, label)

        encoder = Encoder(in_channels=3,
                            output_size=512).cuda()
        out = encoder(img)