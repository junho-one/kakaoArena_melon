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
import metrics
import data_utils
import logger

parser = argparse.ArgumentParser()
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
parser.add_argument("--top_k",
	type=int,
	default=10,
	help="compute metrics@top_k")
parser.add_argument("--factor_num",
	type=int,
	default=32,
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
	type=int,
	default=3,
	help="number of layers in MLP model")
parser.add_argument("--num_ng",
	type=int,
	default=4,
	help="sample negative items for training")
parser.add_argument("--out",
	default=True,
	help="save model or not")
parser.add_argument("--gpu",
	type=str,
	default="0",
	help="gpu card ID")

parser.add_argument("--dataset",
	type=str,
	default="valid",
	help="train for 'valid' or 'test'")


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

if not os.path.exists(os.path.dirname(config.train_log)):
	os.mkdir(os.path.dirname(config.train_log))

print("dataset :",args.dataset)

if __name__ == "__main__":
	############################## PREPARE DATASET ##########################
	print("Start train.py")
	print(args.batch_size)

	train_data, test_question, test_answer, user_num ,item_num, train_mat, user_map, item_map = data_utils.load_all(args.dataset)

	train_dataset = data_utils.NCFData(
			train_data, item_num, train_mat, args.num_ng, True, user_map, item_map)
	test_dataset = data_utils.NCFData(
			test_question, item_num, train_mat, args.num_ng, True, user_map, item_map)

	train_loader = data.DataLoader(train_dataset,
			batch_size=args.batch_size, shuffle=True, num_workers=4)

	test_loader = data.DataLoader(test_dataset,
			batch_size=args.batch_size, shuffle=False, num_workers=0)


	########################### CREATE MODEL #################################
	if config.model == 'NeuMF-pre':
		assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
		assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
		GMF_model = torch.load(config.GMF_model_path)
		MLP_model = torch.load(config.MLP_model_path)
	else:
		GMF_model = None
		MLP_model = None

	model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
							args.dropout, config.model, GMF_model, MLP_model)
	model.cuda()
	loss_function = nn.BCEWithLogitsLoss()

	if config.model == 'NeuMF-pre':
		optimizer = optim.SGD(model.parameters(), lr=args.lr)
	else:
		optimizer = optim.Adam(model.parameters(), lr=args.lr)

	print(model)
	########################## TRAINING #####################################
	best_hr = 0

	for epoch in range(args.epochs):
		print("START {}".format(epoch))
		model.train() # Enable dropout (if have).
		start_time = time.time()
		train_loader.dataset.ng_sample_train()
		uinum = 0
		logger.write_log(config.train_log, "strart train {} epoch".format(epoch))
		for user, item, label in train_loader:
			user = user.cuda()
			item = item.cuda()
			label = label.float().cuda()
			model.zero_grad()
			print("ZZZZ")
			prediction = model(user, item)
			loss = loss_function(prediction, label)
			loss.backward()
			optimizer.step()
			uinum += 1

		if args.dataset == 'valid' :
			acc = metrics.accuracy(model, test_loader, args.top_k)

		if args.out:
			if not os.path.exists(config.model_path):
				os.mkdir(config.model_path)
			torch.save(model.state_dict(),
				'{}{}_{}_{}.pth'.format(config.model_path, config.model, args.dataset, epoch))

		elapsed_time = time.time() - start_time

		logger.write_log(config.train_log, "The time elapse of epoch {:03d}".format(epoch) + " is: " +
				time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		logger.write_log(config.train_log, "-------------------------------------")
