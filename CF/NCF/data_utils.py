import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch.utils.data as data

import config
from collections import defaultdict


def load_all(status="valid"):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
        config.train_data,
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    # valid로도 predict할 수도 있으니 이부분도 바꿔야 할
    if status == "valid":
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
        test_data = test_answer.values.tolist()

    elif status == "test":
        test_question = pd.read_csv(
            config.test_question,
            sep='\t', header=None, names=['user', 'item'],
            usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

        train_data = pd.concat([train_data, test_question])
        test_data = test_question.values.tolist()

    user_num = train_data['user'].nunique() + 1
    item_num = train_data['item'].nunique() + 1

    item_unique = train_data['item'].unique()
    item_map = dict()

    for idx, item in enumerate(item_unique):
        item_map[item] = idx

    user_unique = train_data['user'].unique()
    user_map = dict()

    for idx, user in enumerate(user_unique):
        user_map[user] = idx

    train_data = train_data.values.tolist()

    # 매핑해줘야함. 유니크한 개수는 10만인데 최대값은 15만. Embedd layer에서 값을 벗어난다.
    train_data = [(user_map[user], item_map[item]) for user, item in train_data]
    test_data = [(user_map[user], item_map[item]) for user, item in test_data]

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    return train_data, test_data, user_num, item_num, train_mat, user_map, item_map


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
        self.features = None
        self.labels = [1 for _ in range(len(features))]

    # self.features_fill = None

    # def data_mapping(self, data):
    # 	mapped_data = [(self.user_map[user], self.item_map[item]) for user, item in data]
    # 	return mapped_data

    # 이부분 최적화하기기
    # 현재문제, test_questions도 train에 들어가서 여기서 ng 데이터를 생성해버림.
    def ng_sample_train(self):
        assert self.is_training, 'no need to sampling when testing'
        print("start train ng_sample")
        self.features_ng = []
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

    def ng_sample_test(self):
        print("strart test ng_sample")
        test_ng = []
        index = set(self.features_ps)

        for user_id, item_id in self.features_ps:
            for _ in range(self.num_ng):
                item_not = np.random.randint(self.num_item)
                while (user_id, item_not) in index:
                    item_not = np.random.randint(self.num_item)
                test_ng.append((user_id, item_not))

        test_labels_ps = [1 for _ in range(len(self.features_ps))]
        test_labels_ng = [0 for _ in range(len(test_ng))]

        self.features = self.features_ps + test_ng
        self.labels = test_labels_ps + test_labels_ng

    def all_sample_predict(self):
        # self.feture에 있는 question을 가지고 모든 가짓수의 데이터들을 만듬
        unique_plylst = set()
        plylst_dict = defaultdict(list)
        all_features = []
        all_labels = []

        for plylst, song in self.features_ps:
            unique_plylst.add(plylst)
            plylst_dict[plylst].append(song)

        # troch.tensor로 gpu메모리로 사용 가능할까?  torch.cat((a,b) , dim=0)
        for id in unique_plylst:
            label = np.zeros((1, self.num_item))[0]
            label[plylst_dict[id]] = 1

            ply_ind = [id] * self.num_item
            item_ind = list(range(self.num_item))

            feature = list(zip(ply_ind, item_ind))
            label = list(label)

            all_features.extend(feature)
            all_labels.extend(label)

        self.features = all_features
        self.labels = all_labels

        return plylst_dict

    def __len__(self):
        # return (self.num_ng + 1) * len(self.labels)
        # print(self.is_training)
        # if self.is_training :
        # 	return len(self.features_fill)
        # else :
        # 	return len(self.features)
        return len(self.labels)

    def __getitem__(self, idx):
        # features = self.features_fill if self.is_training \
        # 			else self.features
        # labels = self.labels_fill if self.is_training \
        # 			else self.labels
        features = self.features
        labels = self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]

        return user, item, label

