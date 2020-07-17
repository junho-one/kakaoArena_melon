import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import copy

import model
import config
import evaluate
import data_utils


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",
	type=int,
	default=256,
	help="batch size for training")
parser.add_argument("--top_k",
	type=int,
	default=10,
	help="compute metrics@top_k")
parser.add_argument("--gpu",
	type=str,
	default="0",
	help="gpu card ID")
parser.add_argument("--status",
	type=str,
	default="valid",
	help="train for 'valid' or 'test'")

parser.add_argument("--dropout",
	type=float,
	default=0.0,
	help="dropout rate")
parser.add_argument("--factor_num",
	type=int,
	default=32,
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
	type=int,
	default=3,
	help="number of layers in MLP model")


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


train_data, test_data, user_num ,item_num, train_mat, user_map, item_map = data_utils.load_all(args.status)

inv_user_map = {v: k for k, v in user_map.items()}
inv_item_map = {v: k for k, v in item_map.items()}


# train_dataset = data_utils.NCFData(
# 		train_data, item_num, train_mat, 0, True, user_map, item_map)
test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, False, user_map, item_map)


# train_loader = data.DataLoader(train_dataset,
# 		batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=item_num, shuffle=False, num_workers=0)

GMF_model = None
MLP_model = None

# 모델의 초기값 맞춰줘야 하기에 factor_num같은거 인자로 받아서 씀
model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
						args.dropout, config.model, GMF_model, MLP_model)

model.cuda()
model.load_state_dict(torch.load("./models/NeuMF-end_test_0.pth"))

test_loader.dataset.all_sample_predict()
evaluate.predict(model, test_loader, inv_user_map, inv_item_map, top_k=1000)
# 수정
# acc = evaluate.predict(model, test_loader, args.top_k)