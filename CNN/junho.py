from data_utils import melData
import torch
import torch.utils.data as data
data_set = melData("./arena_mel/")

data_loader = data.DataLoader(dataset=data_set,
# for x,y in data_set :
#     print(x.shape)
#     print(y.size())
#     if prevx != x.shape or prevy != y.size() :
#         print(x,y)
#         print(x.shape, y.size())
#         break
#
#     prevy= y.size()
#     prevx = x.shape

# for x, y in data_loader :
#     print(x.shape)
#     print(y.shape)
    # exit()
                              batch_size=100,
                              shuffle=True,
                              num_workers=1)


for x,y in data_loader :
    print(x.shape)