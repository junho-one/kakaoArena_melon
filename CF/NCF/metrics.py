import numpy as np
import torch
import time
from collections import defaultdict
from tqdm import tqdm

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


# def accuracy(model, test_loader):
#     start_time = time.time()
#     all_predictions = []
#     all_labels = []
#
#     for user, item, label in test_loader:
#         user = user.cuda()
#         item = item.cuda()
#
#         predictions = model(user, item)
#         predictions = torch.round(torch.sigmoid(predictions)).cpu().data.numpy()
#         label = np.array(label)
#
#         all_predictions.extend(predictions)
#         all_labels.extend(label)
#
#     end_time = time.time()
#     logger.write_log(config.train_log, "eval time : {}".format(end_time-start_time))
#
#     logger.write_log(config.train_log, "recall : {}".format(recall_score(all_labels,all_predictions)))
#     logger.write_log(config.train_log, "precision : {}".format(precision_score(all_labels, all_predictions)))
#     logger.write_log(config.train_log, "accuracy : {}".format(accuracy_score(all_labels, all_predictions)))


def predict(model, test_loader, test_question, inv_user_map, inv_item_map, top_k=1000):
    predict_dict = defaultdict(list)

    question_dict = defaultdict(list)
    for plylst, song in test_question:
        question_dict[plylst].append(song)

    for user, item, label in test_loader:

        assert len(user.unique()) == 1, "user id is not unique, please check batch size or Dataset getitem"

        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)

        rating, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
            item, indices).cpu().numpy().tolist()

        questions = set(question_dict[user.unique().cpu().numpy().tolist()[0]])
        recommends = set(recommends)
        recommends = recommends - questions

        userid = str(inv_user_map[user.unique().cpu().numpy().tolist()[0]])
        recommends = [str(inv_item_map[item]) for item in recommends]

        predict_dict[userid].append(recommends)

    return predict_dict

# test_loader에서 user가 가질 수 있는 모든 item이 나와서 확률을 다 구한다.
# 상위 1000개를 뽑고, question에 포함되어 있던 데이터는 제거한 뒤 answer과 비교해 몇개가 맞는지 확인한다.
def hit(model, test_loader, test_question, test_answer, top_k=1000):
    cnt = 0
    total_answers = 0
    total_right = 0
    total_recommends = 0
    total_ratio = 0

    total_recall = 0

    answer_dict = defaultdict(list)
    for plylst, song in test_answer:
        answer_dict[plylst].append(song)

    question_dict = defaultdict(list)
    for plylst, song in test_question:
        question_dict[plylst].append(song)

    for user, item, label in tqdm(test_loader):

        user = user.cuda()
        item = item.cuda()

        assert len(user.unique()) == 1, "user id is not unique, please check batch size or Dataset getitem"

        predictions = model(user, item)

        rating, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
            item, indices).cpu().numpy().tolist()
        
        userid = user.unique().cpu().numpy().tolist()[0]

        answers = set(answer_dict[userid])
        questions = set(question_dict[userid])

        recommends = set(recommends)
        recommends = recommends - questions

        rights = answers & recommends

        total_answers += len(answers)
        total_right += len(rights)

        total_recommends += len(recommends)

        total_ratio += len(rights) / len(answers)

        cnt += 1

    logger.write_log(config.pred_log, "total answer : {}".format(total_answers))

    logger.write_log(config.pred_log, "total ratio : {}".format(total_ratio / cnt))
    logger.write_log(config.pred_log, "total right : {}".format(total_right))
    logger.write_log(config.pred_log, "total recoomends : {}".format(total_recommends))
    logger.write_log(config.pred_log, "total recall : {}".format(total_recall/cnt))




