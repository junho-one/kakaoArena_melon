import json
import numpy as np
import torch
import os

from model import Encoder
from data_utils import melData


os.environ["CUDA_VISIBLE_DEVICES"] = "4"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


predictions = json.load(open("../CF/NCF/preds/pred_0.txt"))

data_set = melData("/root/data/arena_mel/", is_training=False)

encoder = Encoder().cuda()
encoder.load_state_dict(torch.load('./models/encoder_0.pth'))

for user, item in predictions.items() :
    user = int(user)
    data_set.make_batch(item, batch_size=512)

    for X in data_set :
        output = encoder(X.cuda())
        print(output[0].size())
