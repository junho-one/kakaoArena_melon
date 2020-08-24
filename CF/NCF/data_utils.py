import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch.utils.data as data
import torch
import config


def load_all(dataset="valid"):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
        config.train_data,
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    # valid로도 predict할 수도 있으니 이부분도 바꿔야 할
    # if dataset == "valid":
    test_question = pd.read_csv(
        config.val_question,
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    test_answer = pd.read_csv(
        config.val_answer,
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    #  테스트셋을 question, answer로 나누고 question은 트레이닝 셋에 넣는다.
    train_data = pd.concat([train_data, test_question])

    # 현재 데이터셋이 valid가 아니라 test용이라면 valid set을 train set에 병합시킨 뒤 학습시킨다.
    if dataset == "test":
        test_question_for_test = pd.read_csv(
            config.test_question,
            sep='\t', header=None, names=['user', 'item'],
            usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

        train_data = pd.concat([train_data, test_question, test_answer, test_question_for_test])
        test_question = test_question_for_test
        test_answer = pd.DataFrame()

    user_num, user_map = _mapped(train_data['user'])
    item_num, item_map = _mapped(train_data['item'])

    train_data = train_data.values.tolist()
    test_question = test_question.values.tolist()
    test_answer = test_answer.values.tolist()

    # 매핑해줘야함. 유니크한 개수는 10만인데 최대값은 15만. Embedd layer에서 값을 벗어난다.
    test_question = [(user_map[user], item_map[item]) for user, item in test_question]
    train_data = [(user_map[user], item_map[item]) for user, item in train_data]
    test_answer = [(user_map[user], item_map[item]) for user, item in test_answer]

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    return train_data, test_question, test_answer, user_num, item_num, train_mat, user_map, item_map


def _mapped(series) :
    series_num = series.nunique()
    series_unique = series.unique()
    series_map = dict()

    for idx, item in enumerate(series_unique):
        series_map[item] = idx

    return series_num, series_map


class NCFData(data.Dataset):
    def __init__(self, features,
                 num_item, train_mat=None, num_ng=0, is_training=None, user_map=None, item_map=None):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.user_map = user_map
        self.item_map = item_map

        self.features = features
        self.labels = [1 for _ in range(len(features))]

    # 이부분 최적화하기기
    # 현재문제, test_questions도 train에 들어가서 여기서 ng 데이터를 생성해버림.
    def ng_sample_train(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        print("start train ng_sample")

        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):  # train에 있는 positive 데이터 하나마다 num_ng개의 negative 데이터를 만든다
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)

                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features = self.features_ps + self.features_ng
        self.labels = labels_ps + labels_ng

    def all_sample_test(self):
        self.features = set()

        for user, item in self.features_ps :
            self.features.add(user)

        self.features = list(self.features)

        print("question's unique user count : ", len(self.features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        if self.is_training :
            user = self.features[idx][0]
            item = self.features[idx][1]
            label = self.labels[idx]

        else :
            user = torch.tensor([self.features[idx]] * self.num_item)
            item = torch.tensor(range(self.num_item))
            label = torch.zeros(self.num_item)

        return user, item, label


