# dataset name 
# dataset = 'm1-1m_5000'
# dataset = 'ml-1m'
dataset = 'melon'
# assert dataset in ['ml-1m', 'pinterest-20']

# model name 
model = 'NeuMF-end'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = './Data/'

train_data = main_path + '{}_train.txt'.format(dataset)
val_question = main_path + '{}_val_question.txt'.format(dataset)
val_answer = main_path + "{}_val_answer.txt".format(dataset)

test_question = main_path + "{}_test.txt".format(dataset)

# test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = './models/'
pred_path = './preds/'
train_log = './logs/train_log.txt'
pred_log =  './logs/pred_log.txt'

GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
