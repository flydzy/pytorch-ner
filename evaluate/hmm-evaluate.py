import sys
sys.path.append("..")

from Advtrans.models.hmm import HMM
from Advtrans.preprocess.preprocess import build_iterator
import torch
import time

# def accuracy(pred_list, tag_list):
#     correct = 0
#     allcount = 0
#     for i in range(len(pred_list)):
#         for j in range(len(pred_list[i])):
#             if pred_list[i][j] == tag_list[i][j]:
#                 correct += 1
#         allcount += len(pred_list[i])
#     return correct / allcount

# # print("Training HMM...")
# train_data_iter = build_iterator('data/en/eng.train', batch_size=128)
# test_data_iter = build_iterator('data/en/eng.train', batch_size=128)
# tag2idx = train_data_iter.dataset.fields['label'].vocab.stoi
# word2idx = train_data_iter.dataset.fields['sent'].vocab.stoi
# hmm_model = HMM(len(tag2idx), len(word2idx))
# 分批次
# for i,batch in enumerate(train_data_iter):
#     hmm_model.train(batch.sent, batch.label)
#     if i % 100 == 0:
#         print("{} batches trained".format(i))
#         pred_list = []
#         tag_list = []
#         for i,batch in enumerate(test_data_iter):
#             tag_list.extend(batch.label.tolist())
#             result = hmm_model.test(batch.sent)
#             pred_list.extend(result)
#             # print(tag_list[1],pred_list[1])
#         print("batch:{}, Accuracy: {}".format(i, accuracy(pred_list, tag_list)))

# 一次性训练
# print("Training HMM...")
# sents = []
# labels = []
# for i, batch in enumerate(train_data_iter):
#     if i % 100 == 0:
#         print("{} batches trained, len of sents {}".format(i, len(sents)))

#     sents.extend([example.sent for example in batch.dataset.examples])
#     labels.extend([example.label for example in batch.dataset.examples])

# hmm_model.train(sents, labels, word2id=word2idx, tag2id=tag2idx)
# # print(sents, labels)

# print("Training finished")

# sents2 = []
# labels2 = []
# for batch in test_data_iter:
#     sents2.extend([example.sent for example in batch.dataset.examples])
#     labels2.extend([example.label for example in batch.dataset.examples])
# result = hmm_model.test(sents2, word2id=word2idx, tag2id=tag2idx)
# print("Accuracy: {}".format(accuracy(result, labels2)))
train_iter = build_iterator('data/en/eng.train', batch_size=32)
valid_iter = build_iterator('data/en/eng.testa', batch_size=32)
test_iter = build_iterator('data/en/eng.testb', batch_size=32)

tag2idx = train_iter.dataset.fields['label'].vocab.stoi
word2idx = train_iter.dataset.fields['sent'].vocab.stoi

def getOriginData(data_iter, part=1):
    sents = []
    labels = []
    for batch in data_iter:
        sents.extend([example.sent for example in batch.dataset.examples])
        labels.extend([example.label for example in batch.dataset.examples])
    return sents[:int(len(sents)*part)], labels[:int(len(labels)*part)]

# 获取原始数据并展开
dataset = {}
dataset['train'] = getOriginData(train_iter,part=0.01)
dataset['valid'] = getOriginData(valid_iter,part=0.01)
dataset['test'] = getOriginData(test_iter,part=0.01)

print("Training CRF...")
model = HMM(len(tag2idx), len(word2idx))
model.train(dataset['train'][0], dataset['train'][1], word2id=word2idx, tag2id=tag2idx)
# 评估模型
def accuracy(pred_list, tag_list):
    correct = 0
    allcount = 0
    for i in range(len(pred_list)):
        for j in range(len(pred_list[i])):
            if pred_list[i][j] == tag_list[i][j]:
                correct += 1
        allcount += len(pred_list[i])
    return correct / allcount

print("Training finished")
pred_list = model.test(dataset['valid'][0],word2id=word2idx, tag2id=tag2idx)
print("Eval Accuracy: {}".format(accuracy(pred_list, dataset['valid'][1])))

# 测试集测试
pred_list = model.test(dataset['test'][0],word2id=word2idx, tag2id=tag2idx)
print("Test Accuracy: {}".format(accuracy(pred_list, dataset['test'][1])))

