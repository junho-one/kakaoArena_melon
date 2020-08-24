import torch.nn as nn

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # batch x 16 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            # nn.MaxPool2d(2, 2)  # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            # nn.Conv2d(128, 256, 3, stride=1, padding=1),  # batch x 64 x 7 x 7
            # nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Linear(in_features=128*3*36, out_features=5000)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # print(out.shape)
        out = out.view(-1, 128*3*36)
        out = self.layer3(out)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Linear(in_features=5000, out_features=128*3*36)
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 2, 1,1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 2, 1,1),  # batch x 16 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),  # batch x 1 x 28 x 28
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(-1,128,3,36)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

