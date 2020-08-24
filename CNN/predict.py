from collections import Counter

import json
import argparse
import torch
import os
import numpy as np
import time
import pandas as pd

from model import Encoder
from glob import glob

def _cosine_similarity(x1, x2):
    x1 = torch.tensor(x1).cuda()
    return (x1 * x2).sum() / ((x1**2).sum()**.5 * (x2**2).sum()**.5)

def get_recommends(user_emb, items,item_id, top=100) :
    similarities = []

    for item in items :
        similarities.append(_cosine_similarity(user_emb,item))

    item_similar_map = list(zip(item_id, similarities))
    similarities = sorted(item_similar_map, key=lambda x :(-x[1],x[0]))
    recommends =  [id for id,sim in similarities[:top]]

    return recommends

def parser_add_argument	( parser ) :
    parser.add_argument("--gpu",
                        type=str,
                        default="5",
                        help="gpu card ID")
    parser.add_argument("--predictions_file",
                        type=str,
                        default="../CF/NCF/preds/pred.txt",
                        help="output file of NCF")
    parser.add_argument("--image_folder",
                        type=str,
                        default="Data/arena_mel/",
                        help="folder in which mel-spectograms are saved")
    parser.add_argument("--pred_path",
                        type=str,
                        default="preds/",
                        help="result is saved in here")
    return parser


def make_image_path(data_path) :
    dir_list = glob(os.path.join(data_path,"*"))
    paths = []
    for dir in dir_list :
        files = glob(os.path.join(dir,"*"))
        paths.extend(files)

    ret = {}
    for path in paths :
        num = path.split("/")[-1].split(".")[0]
        ret[int(num)] = path

    return ret

def Images(paths):
    images = []

    for path in map(int, paths):
        if _check_image_cache(path):
            images.append(_load_image_cache(path))
        else:
            images.append(_load_image(path))

    return images

def make_image_cache(paths):
    cache = {}
    for num, cnt in paths:
        cache[num] = _load_image(num)

    print("LOAD ALL", len(cache))
    return cache

def _load_image(idx):
    minVal = -100
    maxVal = 26.924

    def _MinMaxScale(array):
        return (array - minVal) / (maxVal - minVal)
    image = np.load(image_paths[idx])

    if image.shape[1] != 576:
        # image = _MinMaxScale(np.resize(image, (48,576)))
        image = np.resize(image, (48, 576))

    return np.reshape(image, (1,) + image.shape)

def _load_image_cache(idx):
    return image_cache[idx]

def _check_image_cache(idx):
    if image_cache.get(idx) is None:
        return False
    else:
        return True

def load_question(data_path) :
    question = pd.read_csv(
        data_path,
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    question = question.groupby('user')['item'].apply(list)
    return question.to_dict()

def make_user_embedding(question) :
    embedding = {}
    for user, items in question.items():
        items = torch.tensor(Images(items)).cuda()
        items = encoder(items)

        means = torch.sum(items, dim=0) / items.size()[0]

        embedding[user] = means.tolist()
        del items
        del means

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser = parser_add_argument( parser )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #################### LOAD MODEL ##############################
    encoder = Encoder().cuda()
    encoder.load_state_dict(torch.load('./models/encoder_0.pth'))

    ################## PREPARE DATA ####################
    predictions = json.load(open(args.predictions_file))

    # predictions file에 있는 image들은 중복된 값이 많아 똑같은 값을 여러번 부르면 시간이 오래걸린다.
    # 그래서 미리 불러와 image_cache라는 딕셔너리에 저장해둔다.
    image_paths = make_image_path(args.image_folder)
    counter = Counter()
    for user, items in predictions.items() :
        counter.update(items[0])

    image_cache = make_image_cache(counter.most_common())

    # user 즉, plylst에 들어 있는 song들의 임베딩 벡터값을 구한 뒤 평균을 취해 user의 임베딩 벡터를 만든다.
    test_question = load_question("../CF/NCF/Data/melon_val_question.txt")
    user_emb = make_user_embedding(test_question)

    ################### PREDICT #########################
    if not os.path.exists(args.pred_path):
        os.mkdir(args.pred_path)

    answer = {}
    for user_id, item_ids in predictions.items() :
        item_ids = item_ids[0]

        items = torch.tensor(Images(item_ids)).cuda()
        items = encoder(items)

        recommends= get_recommends(user_emb[user_id],items,item_ids,top=100)

        del items
        answer[user_id] = recommends

    with open( os.path.join(args.pred_path, "pred_100.txt"), 'w') as file:
        file.write(json.dumps(answer))

    elapsed_time = time.time() - start_time
    print("The time elapse is {}".format(time.strftime("%H: %M: %S", time.gmtime(elapsed_time))))




