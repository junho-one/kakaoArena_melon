import numpy as np
import torch
import time
from collections import defaultdict

from sklearn.metrics import  accuracy_score, precision_score, recall_score
import logger
import config

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def accuracy(model, test_loader, top_k):

	start_time = time.time()
	all_predictions = []
	all_labels = []

	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)
		predictions = torch.round(torch.sigmoid(predictions)).cpu().data.numpy()
		label = np.array(label)

		all_predictions.extend(predictions)
		all_labels.extend(label)


	end_time = time.time()
	logger.write_log(config.train_log, "eval time : {}".format(end_time-start_time))

	logger.write_log(config.train_log, "recall : {}".format(recall_score(all_predictions, all_labels)))
	logger.write_log(config.train_log, "precision : {}".format(precision_score(all_predictions, all_labels)))
	logger.write_log(config.train_log, "accuracy : {}".format(accuracy_score(all_predictions, all_labels)))


# 1개 답인데 10개 뽑음 -> 100개 답이면 1000개?
# 그럼 내꺼에서는 100개 뽑아내고 NDCG 비교?


def predict(model, test_loader, inv_user_map, inv_item_map, top_k=1000):
	start_time = time.time()
	# 한 유저당 모든 item 중에 탑 1000개를 뽑기?
	predict_dict = defaultdict(list)

	for user, item, label in test_loader:

		if len(user.unique()) > 1 :
			print(user)
			print("ERROR")
			exit()


		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)

		rating, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
			item, indices).cpu().numpy().tolist()

		recommends = set(item[0][predictions > 0].cpu().numpy().tolist())

		userid = inv_user_map[user.unique().cpu().numpy().tolist()[0]]
		recommends = [inv_item_map[item] for item in recommends]

		# predictions.append({"id":userid, "songs":recommends})
		predict_dict[userid].append(recommends)

	end_time = time.time()
	print("predict time : ", end_time-start_time)
	return predict_dict


def hit(model, test_loader, test_question, test_answer, top_k=100):
	# 한 유저당 모든 item 중에 탑 1000개를 뽑기?
	cnt = 0
	total_answers = 0
	total_right = 0
	total_recommends = 0
	total_ratio = 0

	total_recommends2 = 0
	total_right2 = 0

	answer_dict = defaultdict(list)
	for plylst, song in test_answer:
		answer_dict[plylst].append(song)

	question_dict = defaultdict(list)
	for plylst, song in test_question:
		question_dict[plylst].append(song)

	for user, item, label in test_loader:

		user = user.cuda()
		item = item.cuda()

		if len(user.unique()) > 1:
			print(user)
			print(user.shape)
			print("EXIT")
			exit()

		predictions = model(user,item)

		rating, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
			item, indices).cpu().numpy().tolist()

		userid = user.unique().cpu().numpy().tolist()[0]

		answers = set(answer_dict[userid])
		questions = set(question_dict[userid])

		recommends = set(recommends)
		rights = answers & recommends
		recommends2 = set(item[0][predictions>0].cpu().numpy().tolist())
		rigths2 = answers & recommends2

		total_answers += len(answers - questions)
		total_right += len(rights - questions)
		total_right2 += len(rigths2 - questions)
		total_recommends += len(recommends - questions)
		total_recommends2 += len(recommends2 - questions)

		total_ratio += len(rights) / len(answers)
		cnt += 1

	logger.write_log(config.pred_log, "total ratio : {}".format(total_ratio / cnt))
	logger.write_log(config.pred_log, "total right : {}".format(total_right))
	logger.write_log(config.pred_log, "total answer : {}".format(total_answers))
	logger.write_log(config.pred_log, "total recoomends : {}".format(total_recommends))
	logger.write_log(config.pred_log, "total right2 : {}".format(total_right2))
	logger.write_log(config.pred_log, "total recommends2 : {}".format(total_recommends2))

	# 실제 프레딕트 떄는 트레인에 있떤 데이터들은 recommends안에 안들어가게 처리 해줘야 함


# def hit(model, test_loader, test_answer, top_k=1000):
# 	# 한 유저당 모든 item 중에 탑 1000개를 뽑기?
# 	start_time = time.time()
#
# 	answer_dict = defaultdict(list)
# 	total = 0
# 	total_right = 0
# 	total_answer = 0
# 	for plylst, song in test_answer:
# 		answer_dict[plylst].append(song)
# 	print(len(test_loader))
# 	cnt = 0
# 	total_right2 = 0
#
# 	for user, item, label in test_loader:
# 		# print("U",user)
# 		# print("I",item)
# 		cnt += 1
# 		# user = torch.tensor(user).cuda()
# 		# item = torch.tensor(item).cuda()
# 		print("USE", user)
# 		for u,i in zip(user,item) :
# 			u = u.cuda()
# 			i = i.cuda()
# 			print(u)
# 			if len(u.unique()) > 1 :
# 				print(u)
# 				print(i)
# 				print("??")
# 				exit()
#
# 			predictions = model(u,i)
#
# 			rating, indices = torch.topk(predictions, top_k)
# 			recommends = torch.take(
# 				i, indices).cpu().numpy().tolist()
#
# 			rec2_idx = predictions > 0
# 			pred2 =i[rec2_idx]
#
# 			userid = u[0].cpu().numpy().tolist()
#
# 			ans = set(answer_dict[userid])
# 			pred = set(recommends)
# 			right = ans & pred
#
# 			pred2 = set(pred2)
# 			right2 = ans & pred2
#
# 			total_answer += len(ans)
# 			total_right += len(right)
# 			total_right2 += len(right2)
#
# 			total += len(right) / len(ans)
#
# 		e= time.time()
# 		print(cnt," time :",e-s)
# 		s = time.time()
#
# 		print("cnt : {} time : {}".format(cnt,e-s))
# 	end_time = time.time()
# 	print("predict time : ", end_time-start_time)
# 	print("total ratio : ", total / len(answer_dict))
# 	print("total right : ", total_right)
# 	print("total answer : ", total_answer)
# 	print("total right2 : ",total_right2)


# def metrics(model, test_loader, top_k):
# 	HR, NDCG = [], []
# 	start_time = time.time()
#
# 	for user, item, label in test_loader:
#
# 		user = user.cuda()
# 		item = item.cuda()
# 		predictions = model(user, item)
#
#
# 		_, indices = torch.topk(predictions, top_k)
# 		recommends = torch.take(
# 				item, indices).cpu().numpy().tolist()
#
# 		print(label)
# 		print(predictions)
#
# 		# gt_item 즉 한 줄 데이터에서 첫번째 값이 1이니까 그거만 정답이라는 것..
# 		gt_item = item[0].item()
# 		HR.append(hit(gt_item, recommends))
# 		NDCG.append(ndcg(gt_item, recommends))
#
# 	end_time = time.time()
# 	print("time : ", end_time-start_time)
#
# 	return np.mean(HR), np.mean(NDCG)

