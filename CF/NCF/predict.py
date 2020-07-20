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
import json
import logger


if not os.path.exists(os.path.dirname(config.pred_log)):
	os.mkdir(os.path.dirname(config.pred_log))

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",
	type=int,
	default=1,
	help="batch size for training")
parser.add_argument("--top_k",
	type=int,
	default=10,
	help="compute metrics@top_k")
parser.add_argument("--gpu",
	type=str,
	default="0",
	help="gpu card ID")
parser.add_argument("--dataset",
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
parser.add_argument("--out",
	default=True,
	help="save model or not")
parser.add_argument("--epochs",
	type=int,
	default=10,
	help="training epoches")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


if __name__ == "__main__":

	train_data, test_question, test_answer, user_num ,item_num, train_mat, user_map, item_map = data_utils.load_all(args.dataset)

	inv_user_map = {v: k for k, v in user_map.items()}
	inv_item_map = {v: k for k, v in item_map.items()}

	print("TN",item_num)
	print(max(user_map.keys()))
	print(max(user_map.values()))
	print(max(item_map.keys()))
	print(max(item_map.values()))
	print(max(inv_item_map.keys()))
	print(max(inv_item_map.values()))
	print(max(inv_user_map.keys()))
	print(max(inv_user_map.values()))

	# TN
	# 192020
	# 153428
	# 117725
	# 707986
	# 192018
	# 192018
	# 707986
	# 117725
	# 153428

	test_dataset = data_utils.NCFData(
			test_question, item_num, train_mat, 0, False, user_map, item_map)

	test_loader = data.DataLoader(test_dataset,
			batch_size=args.batch_size, shuffle=False, num_workers=0)

	GMF_model = None
	MLP_model = None

	# 모델의 초기값 맞춰줘야 하기에 factor_num같은거 인자로 받아서 씀
	model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
							args.dropout, config.model, GMF_model, MLP_model)
	model.cuda()
	print(model)


	test_loader.dataset.sample_all_user()

	print("start predict")
	for epoch in range(args.epochs) :
		print("START :", epoch)
		start_time = time.time()

		logger.write_log(config.pred_log, "strart predict {} epoch".format(epoch))

		model.load_state_dict(torch.load('{}{}_{}_{}.pth'.format(config.model_path, config.model, args.dataset, epoch)))

		if args.dataset == 'valid' :
			metrics.hit(model, test_loader, test_question, test_answer)
		elif args.dataset == 'test' :
			predictions = metrics.predict(model, test_loader, inv_user_map, inv_item_map)

		if args.dataset == 'test':
			if not os.path.exists(config.pred_path):
				os.mkdir(config.pred_path)

			with open(os.path.join(config.pred_path, "pred_{}.txt".format(epoch)), "w") as fp :
				fp.write(json.dumps(predictions))

		elapsed_time = time.time() - start_time
		logger.write_log(config.pred_log, "The time elapse of epoch {:03d}".format(epoch) + " is: " +
				time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		logger.write_log(config.pred_log, "-------------------------------------")

# evaluate.predict(model, test_loader, inv_user_map, inv_item_map, top_k=1000)
# 수정
# acc = evaluate.predict(model, test_loader, args.top_k)