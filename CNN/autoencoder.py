import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from data_utils import melData


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

batch_size = 256
learning_rate = 0.0002
num_epoch = 31

data_set = melData("/root/data/arena_mel/")

data_loader = data.DataLoader(dataset=data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # batch x 16 x 28 x 28
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # batch x 32 x 28 x 28
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # batch x 32 x 28 x 28
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            # nn.MaxPool2d(2, 2)  # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # batch x 64 x 7 x 7
            nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # batch x 128 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1,1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, 2, 1,1),  # batch x 16 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),  # batch x 1 x 28 x 28
            nn.ReLU()
        )

    def forward(self, x):
        out = x.view(batch_size, 256, 3, 36)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

encoder = Encoder().to(device)
decoder = Decoder().to(device)

# paprameter를 �~O~Y�~K~\�~W~P �~U~Y�~J��~K~\�~B�기�~\~D�~U� 묶�~V��~F~T�~U��~U~\�~K�.
parameters = list(encoder.parameters())+ list(decoder.parameters())

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)


for i in range(num_epoch):
    print("EPOCH : {}".format(i))
    for image, label in data_loader :
        optimizer.zero_grad()
        image = image.to(device)

        output = encoder(image)
        output = decoder(output)
        loss = loss_func(output, image)
        loss.backward()
        optimizer.step()


    if i % 10 == 0 :
        torch.save(encoder.state_dict(),
                   './models/encoder_{}.pth'.format(i))
        torch.save(decoder.state_dict(),
                   './models/decoder_{}.pth'.format(i))

