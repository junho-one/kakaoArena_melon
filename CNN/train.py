from tqdm import tqdm

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

from model import Encoder, Decoder

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

batch_size = 256
learning_rate = 0.0002
num_epoch = 31

data_set = melData("/root/data/arena_mel/")

data_loader = data.DataLoader(dataset=data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

encoder = Encoder().to(device)
decoder = Decoder().to(device)

# paprameter를 동시에 학습시키기위해 묶어놔야한다.
parameters = list(encoder.parameters())+ list(decoder.parameters())

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

for i in range(num_epoch):
    print("EPOCH : {}".format(i))
    cnt = 0
    #for image, label in tqdm(data_loader) :
    for image in data_loader :
        optimizer.zero_grad()
        image = image.to(device)
        global_batch_size = len(image)
        output = encoder(image)
        output = decoder(output)
        loss = loss_func(output, image)
        loss.backward()
        optimizer.step()
        cnt += 1
        if cnt % 100 == 0 :
            print(cnt*batch_size)
    if i % 3 == 0 :
        torch.save(encoder.state_dict(),
                   './models/encoder_{}.pth'.format(i))
        torch.save(decoder.state_dict(),
                   './models/decoder_{}.pth'.format(i))




