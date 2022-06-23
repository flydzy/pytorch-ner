import sys
import torch
from zmq import device
sys.path.append(r'..')
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from Advtrans.models.lstm import LSTM
from Advtrans.preprocess import  build_dataset,SENT,LABEL
import torch.optim as optim 
import numpy as np
from torchtext.legacy.data import Field, Example, Dataset, BucketIterator, Iterator
# from seqeval.metrics import f1_score, classification_report
from sklearn.metrics import classification_report, precision_recall_fscore_support

# 标签评估的警告
import warnings
warnings.filterwarnings("ignore")

BATCH_SZIE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build dataset
train_dataset = build_dataset(r'data/en/eng.train')
test_dataset = build_dataset(r'data/en/eng.testb')
valid_dataset = build_dataset(r'data/en/eng.testa')

# build vocab
SENT.build_vocab(train_dataset, vectors='glove.6B.50d')
LABEL.build_vocab(train_dataset)

# build iterator
train_iter = BucketIterator(
    train_dataset,
    batch_size=BATCH_SZIE,
    device=DEVICE,
    repeat=False,
    shuffle=True,
    sort_key=lambda x: len(x.sent),
    sort_within_batch=True,
    )
valid_iter = BucketIterator(
    valid_dataset,
    batch_size=BATCH_SZIE,
    device=DEVICE,
    repeat=False,
    shuffle=True,
    sort_key=lambda x: len(x.sent),
    sort_within_batch=True,
    )
test_iter = BucketIterator(
    test_dataset,
    batch_size=BATCH_SZIE,
    device=DEVICE,
    repeat=False,
    sort_key=lambda x: len(x.sent),
    sort_within_batch=True,
    )

# test output the  vocab of the three dataset
# print(train_dataset.fields['label'].vocab.stoi)
# print(valid_dataset.fields['label'].vocab.stoi)
# print(test_dataset.fields['label'].vocab.stoi)

# vocab_size 
labels2id = train_dataset.fields['label'].vocab.stoi
id2labels = train_dataset.fields['label'].vocab.itos

# parameters
VOCAB_SZIE = len(train_dataset.fields['sent'].vocab)
HIDDEN_SIZE = 32
OUT_SIZE = len(labels2id)
EPOCHES = 10
EMBEDDING = train_dataset.fields['sent'].vocab.vectors
EMB_SIZE = EMBEDDING.shape[1]

print('hidden_size:',HIDDEN_SIZE, 'embedding_size:', EMB_SIZE, 'epochs:', EPOCHES, 'out_size:', OUT_SIZE,"vocab_size:", VOCAB_SZIE, "embedding:", EMBEDDING.shape)

model = LSTM(VOCAB_SZIE, EMB_SIZE, HIDDEN_SIZE, OUT_SIZE, EMBEDDING)
optimizer = optim.Adam(model.parameters())
loss_fuc = nn.CrossEntropyLoss(ignore_index=1)  # pad was be ignored

label_idxs = [train_dataset.fields['label'].vocab.stoi[l]  for l in id2labels[3:]]


def align_predictions(predictions, golds, id2labels):
    """
    predictions: [Batch*Length Class]
    labels:[Batch*Length]
    """
    preds_list, labels_list = [], []
    for i in range(len(labels)):
        if labels[i] > 1:
            labels_list.append(id2labels[golds[i]])
            preds_list.append(id2labels[predictions[i]])
    return preds_list, labels_list


loss_list = []
for epoch in range(EPOCHES):
    for step,batch in enumerate(train_iter):
        
        labels = batch.label  # 读取到batch中的labels
        labels = labels.view(-1)  # 将batch中的labels中展开成一个向量
        output = model(batch.sent[0], batch.sent[1]).view(-1, OUT_SIZE) # [B*L N]
        loss = loss_fuc(output, labels)  # output [B*L N] labels [B*L]   
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], loss:{:.4f}".format(epoch+1, EPOCHES, step+1, len(train_iter),loss.item()))# 评估训练集的表现
    
    # 每一个epoch都评估模型
    with torch.no_grad():
        # 评估训练集的表现
        train_preds, train_labels = [], []
        for step,batch in enumerate(train_iter):
            labels = batch.label
            labels = labels.view(-1)
            output = model(batch.sent[0], batch.sent[1]).view(-1, OUT_SIZE)
            golds = labels.cpu().numpy()
            predictions = torch.argmax(output, dim=1).cpu().numpy()
            preds_list, labels_list = align_predictions(predictions, golds, id2labels)
            train_preds.extend(preds_list)
            train_labels.extend(labels_list)
        print("train report:", precision_recall_fscore_support(train_labels, train_preds, average='macro'))
        print(label_idxs, id2labels[3:])    
        print(classification_report(train_labels, train_preds, labels=label_idxs, target_names=id2labels[3:]))
        valid_preds, valid_labels = [], []
        for batch in valid_iter:
            output = model(batch.sent[0], batch.sent[1]).view(-1, OUT_SIZE)
            preds = torch.argmax(output, dim=1)
            valid_preds.extend(preds.cpu().numpy())
            valid_labels.extend(batch.label.view(-1).cpu().numpy())
        valid_preds, valid_labels = align_predictions(valid_preds, valid_labels, id2labels)
        valid_f1 = precision_recall_fscore_support(valid_labels, valid_preds, average='macro')
        print('valid_score:', valid_f1)
        print(classification_report(valid_labels, valid_preds))
        # 评估测试集的表现
        test_preds, test_labels = [], []
        for batch in test_iter:
            output = model(batch.sent[0], batch.sent[1]).view(-1, OUT_SIZE)
            preds = torch.argmax(output, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch.label.view(-1).cpu().numpy())
        test_preds, test_labels = align_predictions(test_preds, test_labels, id2labels)
        test_f1 = precision_recall_fscore_support(test_labels, test_preds, average='macro')
        print('test_score:', test_f1)
        print(classification_report(test_labels, test_preds))

# 最终的测试集的表现
with torch.no_grad():
    test_preds, test_labels = [], []
    for batch in test_iter:
        output = model(batch.sent[0], batch.sent[1]).view(-1, OUT_SIZE)
        preds = torch.argmax(output, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(batch.label.view(-1).cpu().numpy())
    test_preds, test_labels = align_predictions(test_preds, test_labels, id2labels)
    test_f1 = precision_recall_fscore_support(test_labels, test_preds, average='macro')
    print('test_score:', test_f1)
    print(classification_report(test_labels, test_preds))

sns.lineplot(range(len(loss_list)), loss_list)
plt.savefig('imgs/train_loss_'+ str(BATCH_SZIE) + '.png')