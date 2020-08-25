# kakaoArena melon playlist contiunation

이 대회의 목적은 플레이리스트에 수록된 곡과 태그의 정보가 일부만 주어질 때, 나머지 주어지지 않은 플레이리스트의 곡들과 태그를 예측하는 것이다.
해당 대회에서는 주어진 플레이리스트 정보를 통해 원래 플레이리스트에 수록되어 있을 것이라 생각되는 곡 100개와 태그 10개를 예측해 점수를 매긴다.

자세한 내용은 [해당 사이트](https://arena.kakao.com/c/7)를 참고하길 바랍니다.
또한 필요한 데이터는 [이 사이트](https://arena.kakao.com/c/7/data) 에서 다운로드 할 수 있습니다.


# NCF-AutoEncoder recommendation system

위 대회 참가를 목적으로 일부 플레이리스트의 정보가 들어왔을 때, 주어지지 않은 플레이리스트에 수록된 곡을 예측하는 추천 시스템을 개발했습니다. 

11만개의 플레이리스트와 곡의 개수가 70만개에 달하는 방대한 데이터를 다뤄야 하기에 모든 플레이리스트와 곡들간의 유사도를 구하기 위해서는 긴 시간이 소요됩니다. 
시간을 단축시키기 위해서 **Neural Collaborative Filtering(NCF)을 통해 해당 플레이리스트에 속할 가능성이 높은 후보곡들을 추려냈습니다.**

**그 후 플레이리스트와 후보곡들간의 유사도를 구하여 유사도가 가장 높은 100개의 곡을 추천하는 시스템입니다.**
이때 플레이리스트와 곡들간의 유사도를 구하기 위해 해당 곡의 mel-spectogram을 Convolutional Autoencoder와 Cosine Similarity가 사용됩니다.

모든 곡의 mel-spectogram을 학습 데이터로 사용해 Convolutional Autoencoder를 학습시킨 뒤 Encoder 부분만 떼어내어 Embedding model로 사용합니다. 
mel-spectogram이 Encoder를 통과하면 5000차원의 Embedding Vector로 변환되고, 두 Vector간의 Cosine similarity를 통해 유사도를 계산합니다.
이때 플레이리스트의 Embedding Vector는 플레이리스트에 수록된 곡들의 Embedding Vector의 평균값이 됩니다.



# 사용법

## 1. NCF (Collaborative Filtering)

### 1.1 제공된 데이터 전처리 
원본 데이터로부터 학습에 필요한 {"플레이리스트 ID" : ["노래(1)ID", "노래(2)ID" ... "노래(n)ID"]} 형태의 데이터로 변환합니다.
이때 cold start가 우려되는 데이터는 학습에 방해가 될 것이라고 판단하여 제거한 뒤 학습셋을 구축했습니다.

데이터 제거를 위한 임계값으로 user boundary, item boundary 사용합니다.
* user boundary : 노래를 user boundary개 보다 적게 포함하고 있는 플레이리스트는 제거하고,  
* item boundary : 플레이리스트에 item boundary개 보다 적게 포함되어 있는 노래는 제거합니다.
```
python prepareData.py
--data_folder Data/ 
--user_boundary 2 
--item_boundary 5
```

아웃풋 데이터
* melon_train.txt : 학습 데이터셋
* melon_val.txt : 평가 데이터셋의 question에 해당한다. 이 데이터를 이용해 평가 데이터셋에서 주어지지 않은 곡들(answer)을 예측한다.
* melon_test.txt : 결과 데이터셋의 question에 해당한다. 이 데이터를 이용해 결과 데이터셋에서 주어지지 않은 곡들(answer)을 예측한다.



### 1.2 모델 학습
NCF 모델을 불러와 학습시킨다. 
num_ng는 negative sample의 개수로 데이터 준비 과정에서 무작위로 만들어낸다. 

dataset이 valid일 때는 melon_train.txt와 melon_val.txt를 train set으로 이용하여 모델을 학습시키고,
test일 때는 melon_train.txt, melon_val.txt, melon_test.txt를 이용하여 학습시킨다.

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

* 추후 answer 데이터가 공개될 것을 고려하여 코드를 작성해놓았습니다.
config.py에 있는 val_answer 변수에 valid용 정답 데이터를 넣으면 됩니다.

### 1.3 후보군 예측
학습된 모델을 불러와 예측한다.

top_k 파라미터를 통해 logit의 결과가 가장 높은 top_k개의 데이터만 최종 후보군으로 선택합니다
이때 높은 logit을 가져 후보군으로 선정되었지만, question에 속해있는 곡들은 제외시켰습니다.

batch_size를 1만 해도 model에 들어갈 때는 (user,item) 쌍이 item 개수만큼 들어간다. 여기선 대략 30만

```
python predict.py 
--batch_size 1 
--dataset valid 
--factor_num 32 
--gpu 4 
--top_k 1000
```



## 2. Convolutional Autoencoder (content based filtering)

### 2.1 훈련
```
1. python train.py
--image_folder Data/arena_mel
--batch_size 256
--lr 0.001
--epochs 10
```
CNN 기반의 오토인코더를 사용하여 인코더를 학습시킨다.
이때 인코더의 출력은 5000 차원의 벡터(임베딩 벡터)가 된다.
학습 전에 카카오 아레나 사이트에 접속하여 mel spectogram 이미지 파일을 다운받은 후 CNN/Data 폴더에 넣어줘야 한다.

### 2.2 예측
```
python predict.py
--predictions_file ../CF/NCF/preds/pred.txt
--image_folder Data/arena_mel
--pred_path preds/
```
user의 embedding vector는 plylst에 속한 song들의 embedding vector의 평균값이다.
이 user의 임베딩 벡터와 predictions에서 추천된 song들의 임베딩벡터간의 유사도를 구한 후 유사도가 높은 100개만을 추려 최종 추천 아이템으로 선택한다.
