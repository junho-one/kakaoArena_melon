from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import os

from data_utils import melData
from model import Encoder, Decoder


def parser_add_argument	( parser ) :
    parser.add_argument("--image_folder",
        type=str,
        default="Data/arena_mel/",
        help="folder in which mel-spectograms are saved")
    parser.add_argument("--lr",
        type=float,
        default=0.001,
        help="learning rate")
    parser.add_argument("--dropout",
        type=float,
        default=0.0,
        help="dropout rate")
    parser.add_argument("--batch_size",
        type=int,
        default=256,
        help="batch size for training")
    parser.add_argument("--epochs",
        type=int,
        default=10,
        help="training epoches")
    parser.add_argument("--gpu",
        type=str,
        default="0",
        help="gpu card ID")
    return parser



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = parser_add_argument( parser )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    data_set = melData(args.image_folder)
    data_loader = data.DataLoader(dataset=data_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    encoder.train()
    decoder.train()

    # paprameter를 동시에 학습시키기위해 묶어놔야한다.
    parameters = list(encoder.parameters())+ list(decoder.parameters())

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    for i in range(args.epochs):
        print("EPOCH : {}".format(i))

        for image in tqdm(data_loader) :
            optimizer.zero_grad()
            image = image.to(device)
            global_batch_size = len(image)
            output = encoder(image)
            output = decoder(output)

            loss = loss_func(output, image)
            loss.backward()
            optimizer.step()

        if i % 3 == 0 :
            torch.save(encoder.state_dict(),
                       './models/encoder_{}.pth'.format(i))
            torch.save(decoder.state_dict(),
                       './models/decoder_{}.pth'.format(i))
