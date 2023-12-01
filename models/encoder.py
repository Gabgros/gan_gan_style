import torch
import legacy
import dnnlib
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, in_channels, output_size):
        super(Encoder, self).__init__()
        # Nb of filters for each layer
        base_nb_filters = 32
        self.layer_1_kernel_size = base_nb_filters
        self.layer_2_kernel_size = base_nb_filters * 2
        self.layer_3_kernel_size = base_nb_filters * 4
        self.layer_4_kernel_size = base_nb_filters * 8
        self.layer_5_kernel_size = base_nb_filters * 4
        self.layer_6_kernel_size = base_nb_filters * 2
        self.layer_7_kernel_size = base_nb_filters
        # FC Layers
        self.convtoFC = 8 * 8 * self.layer_7_kernel_size
        self.fc_1_size = 1024
        self.output = 512

        # Convolutional layers
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, self.layer_1_kernel_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(self.layer_1_kernel_size),
            nn.Conv2d(self.layer_1_kernel_size, self.layer_2_kernel_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(self.layer_2_kernel_size),
            nn.Conv2d(self.layer_2_kernel_size, self.layer_3_kernel_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.BatchNorm2d(self.layer_3_kernel_size),
            nn.Conv2d(self.layer_3_kernel_size, self.layer_4_kernel_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.BatchNorm2d(self.layer_4_kernel_size),
            nn.Conv2d(self.layer_4_kernel_size, self.layer_5_kernel_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer6 = nn.Sequential(
            nn.BatchNorm2d(self.layer_5_kernel_size),
            nn.Conv2d(self.layer_5_kernel_size, self.layer_6_kernel_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU())
        self.layer7 = nn.Sequential(
            nn.BatchNorm2d(self.layer_6_kernel_size),
            nn.Conv2d(self.layer_6_kernel_size, self.layer_7_kernel_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU())

        # Linear layers
        self.layer8 = nn.Sequential(
            nn.BatchNorm1d(self.convtoFC),
            nn.Linear(self.convtoFC, self.fc_1_size),
            nn.LeakyReLU())
        self.layer9 = nn.Sequential(
            nn.BatchNorm1d(self.fc_1_size),
            nn.Linear(self.fc_1_size, self.output),
            nn.LeakyReLU())
        self.layer10 = nn.Linear(self.output, output_size)

        # Group layers by kind
        self.convlayers = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4,
                                        self.layer5, self.layer6, self.layer7)
        self.fclayers = nn.Sequential(self.layer8, self.layer9, self.layer10)

    def forward(self, x):
        output = self.convlayers(x)
        output = output.reshape(output.size(0), -1)
        output = self.fclayers(output)
        return output
