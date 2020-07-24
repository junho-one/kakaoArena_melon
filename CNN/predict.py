import json
import numpy as np
import torch
import os
import time
from model import Encoder
from data_utils import melData


def cosine_similarity(x1, x2):
    return (x1 * x2).sum() / ((x1**2).sum()**.5 * (x2**2).sum()**.5)

def get_recommends(user, items,item_id, top=100) :
    similarities = []
    
    for item in items :
        similarities.append(cosine_similarity(user,item))
    
    item_similar_map = list(zip(item_id, similarities))

    similarities = sorted(item_similar_map, key=lambda x :(-x[1],x[0]))
    
    recommends =  [id for id,sim in similarities[:top]]
    
    return recommends
    

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


start_time = time.time()

predictions = json.load(open("../CF/NCF/preds/pred_4.txt"))
data_set = melData("/root/data/arena_mel/", is_training=False)

imageset = set()
for user,item in predictions.items() :
    imageset.add(user)
    imageset = imageset | set(item[0])
print("START")
zx=time.time()
data_set.load_all_image(list(imageset))
print("END : ",time.time()-zx)



encoder = Encoder().cuda()
encoder.load_state_dict(torch.load('./models/encoder_0.pth'))
answer = {}
cnt = 0
for user_id, item_ids in predictions.items() :
    cnt+=1
    print("USER ID : {}, {}".format(user_id,cnt))
    a = time.time()
    item_ids = item_ids[0]
    
    for _ in range(100000):
        user = data_set.make_user(user_id)
    z = time.time()
    print("0 :",z-a)
    for _ in range(100):
        data_set.make_batch(item_ids, batch_size=256)
    print("1 :",time.time()-z)
    zz = time.time()
    for _ in range(100):
        data_set.make_batch_v2(item_ids, batch_size=256)
    print("2 :",time.time()-zz)
    exit()
    b = time.time()
    user = encoder(user.cuda())
    c = time.time()
    print("2 :",c-b)
    items = torch.tensor([]).cuda()
    for X in data_set :
        items = torch.cat([items, encoder(X.to(device))])
        print("T",items.shape)
    d = time.time()
    print("3 :",d-c)
    recommends= get_recommends(user,items,item_ids,top=75)
    #print(recommends[:10])
    answer[user] = recommends
    e = time.time()
    print("4 :",e-d)

elapsed_time = time.time() - start_time
print("The time elapse is {}".format(time.strftime("%H: %M: %S", time.gmtime(elapsed_time))))


with open(os.path.join("./pred_top75_.txt"), "w") as fp :
    fp.write(json.dumps(answer))
