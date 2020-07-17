import numpy as np
import torch
import time

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []
	start_time = time.time()

	count = 0
	total_acc = 0

	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()
		predictions = model(user, item)

		# _, indices = torch.topk(predictions, top_k)
		# recommends = torch.take(
		# 		item, indices).cpu().numpy().tolist()

		predictions = torch.round(predictions).cpu().data.numpy()
		label = np.array(label)

		answer = predictions == label
		acc = float( np.sum(answer) * 100 / len(answer) )

		total_acc += acc
		count += 1

	end_time = time.time()
	print("eval time : ", end_time-start_time)
	print("accuract :",total_acc / count)

	return acc

# 1개 답인데 10개 뽑음 -> 100개 답이면 1000개?
# 그럼 내꺼에서는 100개 뽑아내고 NDCG 비교?


def predict(model, test_loader, inv_user_map, inv_item_map, top_k=100):
	start_time = time.time()
	# 한 유저당 모든 item 중에 탑 1000개를 뽑기?

	for user, item, label in test_loader:

		user = user.cuda()
		item = item.cuda()
		predictions = model(user, item)

		rating, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
			item, indices).cpu().numpy().tolist()

		userid = user[0].cpu().numpy().tolist()[0]

		print("reccomed of {}".format(inv_user_map[userid]))
		for idx, rec in enumerate(recommends,start=1) :
			print("{} : {}".format(idx, inv_item_map[rec]))



	end_time = time.time()
	print("predict time : ", end_time-start_time)



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

