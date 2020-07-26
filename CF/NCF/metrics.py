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


def accuracy(model,test_question, test_loader, top_k):

                start_time = time.time()
                all_predictions = []
                all_labels = []
                disappeared_ans = 0
                for user, item, label in test_loader:
                                user = user.cuda()
                                item = item.cuda()

                                predictions = model(user, item)
                                predictions = torch.round(torch.sigmoid(predictions)).cpu().data.numpy()
                                label = np.array(label)

                                predictions = list( set(predictions) - set(test_question[user]))
                                print(len(predictions))
                                all_predictions.extend(predictions)
                                all_labels.extend(label)


                end_time = time.time()
                logger.write_log(config.train_log, "eval time : {}".format(end_time-start_time))

                logger.write_log(config.train_log, "recall : {}".format(recall_score(all_labels,all_predictions)))
                logger.write_log(config.train_log, "precision : {}".format(precision_score(all_labels, all_predictions)))
                logger.write_log(config.train_log, "accuracy : {}".format(accuracy_score(all_labels, all_predictions)))


def predict(model, test_loader, inv_user_map, inv_item_map, top_k=1000):
    start_time = time.time()
    # �~U~\ �~\| �| ~@�~K� 모�~S|  item ��~Q�~W~P �~C~Q 1000��~\를 ��~Q기?
    predict_dict = defaultdict(list)

    for user, item, label in test_loader:

        if len(user.unique()) > 1:
            print(user)
            print("ERROR")
            exit()

        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)

        rating, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
            item, indices).cpu().numpy().tolist()

        # recommends = set(item[0][predictions > 0].cpu().numpy().tolist())

        userid = str(inv_user_map[user.unique().cpu().numpy().tolist()[0]])
        # print(recommends)
        recommends = [str(inv_item_map[item]) for item in recommends]

        # predictions.append({"id":userid, "songs":recommends})
        predict_dict[userid].append(recommends)

    end_time = time.time()
    print("predict time : ", end_time - start_time)
    return predict_dict

def hit(model, test_loader, test_question, test_answer, top_k=1000):
    # �~U~\ �~\| �| ~@�~K� 모�~S|  item ��~Q�~W~P �~C~Q 1000��~\를 ��~Q기?
    cnt = 0
    total_answers = 0
    total_right = 0
    total_recommends = 0
    total_ratio = 0

    total_recommends2 = 0
    total_right2 = 0
    total_ratio2 = 0

    total_recommends3 = 0
    total_right3 = 0
    total_ratio3 = 0

    total_recall = 0
    total_recall2 = 0

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

        predictions = model(user, item)

        rating, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
            item, indices).cpu().numpy().tolist()

        rating, indices = torch.topk(predictions, 10000)
        recommends3 = torch.take(
            item, indices).cpu().numpy().tolist()
        
        userid = user.unique().cpu().numpy().tolist()[0]

        answers = set(answer_dict[userid])
        questions = set(question_dict[userid])
        recommends = set(recommends)
        recommends2 = set(item[0][predictions > 0].cpu().numpy().tolist())
        recommends3 = set(recommends)
        recommends = recommends - questions
        recommends2 = recommends2 - questions
        recommends3 = recommends3 - questions

        rights = answers & recommends
        rights2 = answers & recommends2

        rights3 = answers & recommends3
        total_answers += len(answers)
        total_right += len(rights)
        total_right2 += len(rights2)
        total_right3 += len(rights3)
        
        total_recommends += len(recommends)
        total_recommends2 += len(recommends2)
        total_recommends3 += len(recommends3)

        total_ratio += len(rights) / len(answers)
        total_ratio2 += len(rights2) / len(answers)
        total_ratio3 += len(rights3) / len(answers)
        

        # total_recall += recall_score(list(label),list(recommends))
        # total_recall2 += recall_score(list(label),list(recommends2))

        cnt += 1

    logger.write_log(config.pred_log, "total answer : {}".format(total_answers))

    logger.write_log(config.pred_log, "total ratio : {}".format(total_ratio / cnt))
    logger.write_log(config.pred_log, "total right : {}".format(total_right))
    logger.write_log(config.pred_log, "total recoomends : {}".format(total_recommends))
    logger.write_log(config.pred_log, "total recall : {}".format(total_recall/cnt))

    logger.write_log(config.pred_log, "total ratio2 : {}".format(total_ratio2 / cnt))
    logger.write_log(config.pred_log, "total right2 : {}".format(total_right2))
    logger.write_log(config.pred_log, "total recommends2 : {}".format(total_recommends2))
    logger.write_log(config.pred_log, "total recall2 : {}".format(total_recall2/cnt))


