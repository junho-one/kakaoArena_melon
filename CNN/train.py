import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.init
import torch.utils.data as data
import numpy as np
import os

from model import CNN
from data_utils import melData





os.environ["CUDA_VISIBLE_DEVICES"] = '4'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)



if __name__ == "__main__":

    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100

    data_set = melData("./arena_mel/")

    data_loader = data.DataLoader(dataset=data_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1)

    # model = CNN().to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #
    # total_batch = len(data_loader)
    # print('총 배치의 수 : {}, 배치사이즈: {}'.format(total_batch, batch_size))


    for epoch in range(training_epochs):
        avg_cost = 0

        for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            X = X.to(device)
            Y = Y.to(device)
            print(Y)

            # optimizer.zero_grad()
            # hypothesis = model(X)
            # cost = criterion(hypothesis, Y)
            # cost.backward()
            # optimizer.step()
            #
            # avg_cost += cost / total_batch

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))



# with torch.no_grad():
#     X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
#     Y_test = mnist_test.test_labels.to(device)
#
#     prediction = model(X_test)
#     correct_prediction = torch.argmax(prediction, 1) == Y_test
#     accuracy = correct_prediction.float().mean()
#     print('Accuracy:', accuracy.item())
