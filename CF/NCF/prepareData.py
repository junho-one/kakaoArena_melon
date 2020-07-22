import numpy as np
import pandas as pd
import os
import random

from collections import defaultdict
import random
import math

class DataProcessor :

    def __init__(self,dir_name, train_fn, test_fn, valid_fn, ply_b, song_b):

        self.train = self.load_json(dir_name, train_fn)
        self.test = self.load_json(dir_name, test_fn)
        self.valid = self.load_json(dir_name, valid_fn)

        self.directory = dir_name
        self.train_fn = train_fn
        self.test_fn = test_fn
        self.valid_fn = valid_fn

        self.plylst_boundary = ply_b
        self.song_boundary = song_b

    def load_json(self, dir_name, file_name) :

        json_path = os.path.join(dir_name, file_name)

        data = pd.read_json(json_path, typ='frame')
        plylst_song_map = data[['id', 'songs']]

        plylst_song_map_unnest = np.dstack(
            (
                np.repeat(plylst_song_map.id.values, list(map(len, plylst_song_map.songs))),
                np.concatenate(plylst_song_map.songs.values)
            )
        )

        plylst_song_map = pd.DataFrame(data=plylst_song_map_unnest[0], columns=plylst_song_map.columns)
        plylst_song_map['id'] = plylst_song_map['id'].astype(str)
        plylst_song_map['songs'] = plylst_song_map['songs'].astype(str)
        del plylst_song_map_unnest

        plylst_song_map['id'] = plylst_song_map['id'].astype(float)
        plylst_song_map['id'] = plylst_song_map['id'].astype(int)
        plylst_song_map['songs'] = plylst_song_map['songs'].astype(float)
        plylst_song_map['songs'] = plylst_song_map['songs'].astype(int)

        return plylst_song_map


    def removeFewData(self, user_boundary, song_boundary) :
        print("=> remove song and plylst which appears under the boundary in train set\n")
        print("Before plylst nunique:{}, song nunique:{}".format(self.train['id'].nunique(),
                                                                 self.train['songs'].nunique()))
        # 음악이 boundary개 이하를 갖고 있는 플레이리스트를 지움
        # 플레이리스트 boundary개 이하 들어가는 노래는 지움

        plylst_cnt = pd.DataFrame(self.train['id'].value_counts())
        plylst_cnt = plylst_cnt.rename(columns={"id": "plylst_cnt"})
        plylst_cnt = plylst_cnt.reset_index()

        song_cnt = pd.DataFrame(self.train['songs'].value_counts())
        song_cnt = song_cnt.rename(columns={"songs": "song_cnt"})
        song_cnt = song_cnt.reset_index()

        self.train = self.train.merge(plylst_cnt, left_on='id', right_on='index', how="left")

        self.train = self.train.merge(song_cnt, left_on='songs', right_on='index', how='left')

        self.train = self.train[self.train['song_cnt'] >= song_boundary]
        self.train = self.train[self.train['plylst_cnt'] >= user_boundary]

        del self.train['index_x']
        del self.train['index_y']
        del self.train['plylst_cnt']
        del self.train['song_cnt']

        print("After plylst nunique:{}, song nunique:{}\n".format(self.train['id'].nunique(),
                                                                self.train['songs'].nunique()))


    def removeOutOfTrain(self):
        print("=> remove data which is not in train set\n")
        print("valid : Before plylst nunique:{}, song nunique:{}".format(self.valid['id'].nunique(),
                                                                 self.valid['songs'].nunique()))
        print("test : Before plylst nunique:{}, song nunique:{}\n".format(self.test['id'].nunique(),
                                                                 self.test['songs'].nunique()))

        train_song = self.train.set_index(['songs']).index
        valid_song = self.valid.set_index(['songs']).index
        test_song = self.test.set_index(['songs']).index

        valid_mask = valid_song.isin(train_song)
        test_mask = test_song.isin(train_song)

        self.valid = self.valid.loc[valid_mask]
        self.test = self.test.loc[test_mask]

        print("valid : After plylst nunique:{}, song nunique:{}".format(self.valid['id'].nunique(),
                                                                 self.valid['songs'].nunique()))
        print("test : After plylst nunique:{}, song nunique:{}\n".format(self.test['id'].nunique(),
                                                                 self.test['songs'].nunique()))

    def saveData(self, data, fname):
        fname = 'melon_' + os.path.splitext(fname)[0] + '.txt'
        path = os.path.join( self.directory,  fname)

        data.to_csv(path, index=False, header=None, sep="\t")


    def splitValidation(self):
        test_dict = defaultdict(list)
        test_list = self.valid.values.tolist()

        for idx, data in test_list:
            test_dict[idx].append(data)

        test_questions_dict = {}
        test_answers_dict = {}

        # question의 비율
        ratios = [0.1, 0.3, 0.5, 0.75]

        for idx, val in test_dict.items():

            # question에만 plylst와 song이 들어가는 걸 방지하기 위해
            if len(val) > 3  :
                ratio = random.choice(ratios)
                size = math.ceil(len(val) * ratio)

                test_questions_dict[idx] = val[:size]
                test_answers_dict[idx] = val[size:]

            elif len(val) > 1 :
                test_questions_dict[idx] = val[:1]
                test_answers_dict[idx] = val[1:]

        test_questions = []
        for idx, val in test_questions_dict.items():
            for da in val:
                test_questions.append((idx, da))

        test_answers = []
        for idx, val in test_answers_dict.items():
            for da in val:
                test_answers.append((idx, da))

        test_questions = pd.DataFrame(test_questions)
        test_answers = pd.DataFrame(test_answers)

        self.saveData(test_questions, "val_question")
        self.saveData(test_answers, "val_answer")


    def run(self):
        self.removeFewData(self.plylst_boundary, self.song_boundary)
        self.removeOutOfTrain()

        self.splitValidation()

        self.saveData(self.train, self.train_fn)
        self.saveData(self.valid, self.valid_fn)
        self.saveData(self.test, self.test_fn)

dir_path = './Data'
train_fn ='train.json'
test_fn = 'test.json'
valid_fn = 'val.json'
plylst_boundary = 2
song_boundary = 5

DP = DataProcessor(dir_path, train_fn, test_fn, valid_fn,
                   plylst_boundary, song_boundary)
DP.run()
