from data_utils import melData
import torch
import torch.utils.data as data
import numpy as np
data_set = melData("/root/data/arena_mel")

minNum = 99999
maxNum = -99999
total = 0
for i in range(708) :
    minN, maxN, cnt = data_set.dir_max(i)
    print(i, minN, maxN)
    maxNum = max(maxNum, maxN)
    minNum = min(minNum, minN)
    total+=cnt
print("final",minNum, maxNum, total)
exit()
_,prex,prvy = np.ones((1,48,1876)).shape

smallprex = 10000
smallprey = 9999
bigprex= -10000
bigprey= -9999


data_loader = data.DataLoader(dataset=data_set,
                              batch_size=100,
                              shuffle=True,
                              num_workers=0)

maxNum = -1111
minNum = 1000
for x,y in data_loader :
    maxNum = max(x.max(),maxNum)

print(maxNum)

exit()
for x,y in data_set :
    bigprex = max(bigprex, x.max())
    smallprex = min(smallprex, x.min())

print(bigprex, smallprex)
exit()

for x,y in data_set :
    _, xshape, yshape = x.shape
    smallprex = min(smallprex, xshape)
    bigprex = max(bigprex, xshape)
    smallprey = min(smallprey, yshape)
    bigprey = max(bigprey, yshape)
    if prex != xshape or prvy != yshape :
        print(x.shape)
        break

print(bigprex, smallprex)
print(bigprey, smallprey)

exit()

maxn = -9999999999
minn = 99999999999
for x,y in data_set :
    x=x.reshape(-1)
    maxn = max(maxn, x.max())
    minn = min(minn, x.min())


print(maxn, minn)
