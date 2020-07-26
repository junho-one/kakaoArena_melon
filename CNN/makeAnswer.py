import json
import pandas as pd
import numpy as np

predictions = {}

test = pd.read_json('/root/data/test.json', typ='frame')

ids = test['id'].tolist()
songs = test['songs'].tolist()
tags = test['tags'].tolist()


for user, song, tag in list(zip(ids,songs,tags)) :
    predictions[str(user)] = [song,tag]

with open("./pred_top_100.txt", "r") as fp :
    for line in fp :
        user, item = line.split(":")
        user = user.strip()
        item = item.strip()
        predictions[user][0] = eval(item)

result = []
for user, items in predictions.items() :
    # predictions[user] = predictions[user][:100-counter[user]]
    pred = {}
    pred['id'] = user
    pred['songs'] = items[0]
    pred['tags'] = items[1]
    result.append(pred)

with open("./result_songs.json" , "w") as fp :
    json.dump(result, fp)




