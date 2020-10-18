# kakaoArena melon playlist contiunation

이 대회의 목적은 플레이리스트에 수록된 곡과 태그의 정보가 일부만 주어질 때, 나머지 주어지지 않은 플레이리스트의 곡들과 태그를 예측하는 것입니다.<br/>
해당 대회에서는 주어진 플레이리스트 정보를 통해 원래 플레이리스트에 수록되어 있을 것이라 생각되는 곡 100개와 태그 10개를 예측해 점수를 매깁니다.

자세한 내용은 [해당 사이트](https://arena.kakao.com/c/7)를 참고하길 바랍니다.<br/>
또한 필요한 데이터는 [여기](https://arena.kakao.com/c/7/data) 에서 다운로드 할 수 있습니다.


# NCF-AutoEncoder recommendation system

위 대회 참가를 목적으로 일부 플레이리스트의 정보가 들어왔을 때, 주어지지 않은 플레이리스트에 수록된 곡을 예측하는 추천 시스템을 개발했습니다. 

1. Neural Collaborative Filtering(NCF)을 통해 전체 곡 중에서 해당 플레이리스트에 속할 가능성이 높은 후보곡들을 추려냈습니다.
2. Convolutional AutoEncoder를 통해 뽑아낸 Mel-spectrogram의 Embedding Vector와 코사인 유사도를 이용하여 플레이리스트와 후보곡 중 값이 가장 큰 100개의 곡을 추천합니다.


# 사용법

## 1. NCF (Collaborative Filtering)
[논문 링크](https://arxiv.org/abs/1708.05031)

### 1.1 제공된 데이터 전처리 

```
python prepareData.py
--data_folder Data/ 
--user_boundary 2 
--item_boundary 5
```

원본 데이터로부터 학습에 필요한 {"플레이리스트 ID" : ["노래(1)ID", "노래(2)ID" ... "노래(n)ID"]} 형태의 데이터로 변환합니다.<br/>
이때 cold start가 우려되는 데이터는 학습에 방해가 될 것이라고 판단하여 제거한 뒤 학습셋을 구축했습니다.

데이터 제거를 위한 임계값으로 user boundary, item boundary 사용합니다.
* user boundary : 노래를 user boundary개 보다 적게 포함하고 있는 플레이리스트는 제거하고,  
* item boundary : 플레이리스트에 item boundary개 보다 적게 포함되어 있는 노래는 제거합니다.

아웃풋 데이터
* melon_train.txt : 학습 데이터셋
* melon_val.txt : 평가 데이터셋의 question에 해당합니다. 이 데이터를 이용해 평가 데이터셋에서 주어지지 않은 곡들(answer)을 예측합니다.
* melon_test.txt : 결과 데이터셋의 question에 해당합니다. 이 데이터를 이용해 결과 데이터셋에서 주어지지 않은 곡들(answer)을 예측합니다.



### 1.2 모델 학습

```
python train.py 
--batch_size 4096
--dataset valid
--lr 0.001
--factor_num 32
--gpu 4
--num_ng 20
--epochs 8
```

num_ng : negative sample(해당 플레이리스트에 속하지 않은 곡)의 개수로 데이터 준비 과정에서 무작위로 만들어냅니다. 

dataset : valid일 때는 melon_train.txt와 melon_val.txt를 train set으로 이용하여 모델을 학습하고,<br/>
          test일 때는 melon_train.txt, melon_val.txt, melon_test.txt를 이용하여 학습합니다.
          

### 1.3 후보군 예측
```
python predict.py 
--batch_size 1 
--dataset valid 
--factor_num 32 
--gpu 4 
--top_k 1000
```

batch_size :  한번에 모델에 들어가는 플레이리스트의 개수가 됩니다.<br/>
              1로 하면 하나의 플레이리스트와 모든 곡들간의 쌍이 들어가게 된다. (대략 30만개) 


매 플레이리스트 마다 모든 곡들과의 logit값을 구한 뒤 그 중 top_k만 뽑아냅니다.

top_k 파라미터를 통해 logit의 결과가 가장 높은 top_k개의 데이터만 최종 후보군으로 선택합니다<br/>
이때 높은 logit을 가져 후보군으로 선정되었지만, question에 속해있는 곡들은 제외시켰습니다.



## 2. Convolutional Autoencoder (content based filtering)

### 2.1 훈련
```
1. python train.py
--image_folder Data/arena_mel
--batch_size 256
--lr 0.001
--epochs 10
```

Encoder가 Embedding Model로 사용되고, Encoder의 출력은 5000 차원의 벡터(임베딩 벡터)가 됩니다.

학습 전에 카카오 아레나 사이트에 접속하여 Mel-spectrogram 이미지 파일을 다운받은 후 CNN/Data 폴더에 넣어줘야 합니다.


### 2.2 예측
```
python predict.py
--predictions_file ../CF/NCF/preds/pred.txt
--image_folder Data/arena_mel
--pred_path preds/
```

플레이리스트의 Embedding Vector와 NCF를 통해 추천된 후보곡들의 Embedding Vector간의 유사도를 구한 후 유사도가 높은 곡 100개만을 추려 최종적으로 추천합니다.
> 플레이리스트의 Embedding Vector는 플레이리스트의 미리 보여진 곡들의 Embedding Vector의 평균값

