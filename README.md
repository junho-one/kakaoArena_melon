# kakaoArena melon playlist contiunation

이 콘테스트의 목적은 플레이리스트에 속한 일부 노래들에 대한 정보를 바탕으로 
숨겨져 있는 나머지 곡들을 추정하는 것이다.

사용한 데이터는 [이 사이트](https://arena.kakao.com/c/7) 에서 다운로드 했습니다.

NCF를 사용하여 content based filtering을 하기 위한 후보군을 추린 뒤 
CNN based auto encoder를 통해 학습한 이미지 피쳐를 가지고 노래간 코사인 유사도를 구한다.
유사도가 높으면 같은 플레이리스트에 있을 확률이 높다고 가정하여 해당 노래를 추천한다.

## 1. NCF (collaborative filtering)

### 1.1 제공된 데이터 전처리 
원본 데이터로부터 CF에 필요한 user ID, item ID 매핑 데이터를 구한다.
item을 적게 포함하고 있는 user를 지우기 위한 user boundary와 
user에 적게 포함되어 있는 item을 지우기 위한 item boundary를 통해 
cold start가 우려되는 데이터는 제거한 뒤 학습셋을 구축했다.

```
python prepareData.py
--data_folder Data/ 
--user_boundary 2 
--item_boundary 5
```

아웃풋 데이터는 melon_test.txt
melon_train.txt 
melon_val.txt
melon_val_question.txt
melon_val_answer.txt 가 나온다.

여기서 validation.json을 val_question과 val_answer로 나눴다.
val_question에는 있는 데이터는 플레이리스트에 속한 곡 중에서 보여지는 데이터로써 학습에 사용되고,
val_answer은 train set과 val_question을 학습한 뒤 예측해야 하는 플레이리스트에 숨겨진 곡들이다.


### 1.2 모델 학습
NCF 모델을 불러와 학습시킨다. 
num_ng는 negative sample의 개수로 데이터 준비 과정에서 무작위로 만들어낸다.
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

### 1.3 예측
학습된 모델을 불러와 예측한다.
top_k 파라미터를 통해 logit의 결과가 가장 높은 top_k개의 데이터만 예측 결과로 사용한다.

```
python predict.py 
--batch_size 1 
--dataset valid 
--factor_num 32 
--gpu 4 
--epochs 8
```



## 2. CNN based AutoEncoder (content based filtering)