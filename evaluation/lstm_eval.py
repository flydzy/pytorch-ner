import sys
import torch
from zmq import device
sys.path.append(r'..')
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from Advtrans.models.bilstm import LSTM
from Advtrans.preprocess import  build_dataset
import torch.optim as optim 
import numpy as np
from torchtext.legacy.data import Field, Example, Dataset, BucketIterator, Iterator
# from seqeval.metrics import f1_score, classification_report
from sklearn.metrics import classification_report, precision_recall_fscore_support
sns.set_style("darkgrid")

import warnings
warnings.filterwarnings("ignore")

BATCH_SZIE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

en_dataset = build_dataset(train_path=r"data/en/eng.train", valid_path=r"data/en/eng.testa", test_path=r"data/en/eng.testb")


# build iterator
train_iter = BucketIterator(
    en_dataset['train'],
    batch_size=BATCH_SZIE,
    device=DEVICE,
    repeat=False,
    shuffle=True,
    sort_key=lambda x: len(x.sent),
    sort_within_batch=True,
    )
valid_iter = BucketIterator(
    en_dataset['valid'],
    batch_size=BATCH_SZIE,
    device=DEVICE,
    repeat=False,
    shuffle=True,
    sort_key=lambda x: len(x.sent),
    sort_within_batch=True,
    )
test_iter = BucketIterator(
    en_dataset['test'],
    batch_size=BATCH_SZIE,
    device=DEVICE,
    repeat=False,
    sort_key=lambda x: len(x.sent),
    sort_within_batch=True,
    )

# test output the  vocab of the three dataset
print(en_dataset['train'].fields['label'].vocab.stoi)
print(en_dataset['valid'].fields['label'].vocab.stoi)
print(en_dataset['test'].fields['label'].vocab.stoi)

# vocab_size 
labels2id = en_dataset['train'].fields['label'].vocab.stoi
id2labels = en_dataset['train'].fields['label'].vocab.itos

# parameters
VOCAB_SZIE = len(en_dataset['train'].fields['sent'].vocab)
HIDDEN_SIZE = 256
OUT_SIZE = len(labels2id)
EPOCHES = 10
EMBEDDING = en_dataset['train'].fields['sent'].vocab.vectors
EMB_SIZE = EMBEDDING.shape[1]
ISBIDIRECTIONAL = True

print('hidden_size:',HIDDEN_SIZE, 'embedding_size:', EMB_SIZE, 'epochs:', EPOCHES, 'out_size:', OUT_SIZE,"vocab_size:", VOCAB_SZIE, "embedding:", EMBEDDING.shape)

model = LSTM(VOCAB_SZIE, EMB_SIZE, HIDDEN_SIZE, OUT_SIZE, ISBIDIRECTIONAL ,EMBEDDING)
optimizer = optim.Adam(model.parameters())
loss_fuc = nn.CrossEntropyLoss(ignore_index=0)  # pad was be ignored

tag_name = en_dataset['train'].fields['label'].vocab.itos[3:]
tag_name = sorted(tag_name, key=lambda x: x.split("-")[-1])
label_idxs = [en_dataset['train'].fields['label'].vocab.stoi[l]  for l in tag_name]


def align_predictions(predictions, golds, id2labels):
    """
    predictions: [Batch*Length Class]
    labels:[Batch*Length]
    """
    preds_list, labels_list = [], []
    for i in range(len(labels)):
        if labels[i] > 0:
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
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], loss:{:.4f}".format(epoch+1, EPOCHES, step+1, len(train_iter),loss.item()))# 评估训练集的表现
    loss_list.append(loss.item())
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
        # print(classification_report(train_labels, train_preds, labels=label_idxs, target_names=id2labels[3:]))
        valid_preds, valid_labels = [], []
        for batch in valid_iter:
            output = model(batch.sent[0], batch.sent[1]).view(-1, OUT_SIZE)
            preds = torch.argmax(output, dim=1)
            valid_preds.extend(preds.cpu().numpy())
            valid_labels.extend(batch.label.view(-1).cpu().numpy())
        valid_preds, valid_labels = align_predictions(valid_preds, valid_labels, id2labels)
        valid_f1 = precision_recall_fscore_support(valid_labels, valid_preds, average='macro')
        print('valid_score:', valid_f1)
        # print(classification_report(valid_labels, valid_preds))
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
        # print(classification_report(test_labels, test_preds))

def cout_tags(preds, labels):
    """
    preds: [Batch*Length]
    """
    fix, ax = plt.subplots(1,2,figsize=(10,5))
    sub1 = sns.countplot(preds,ax=ax[0])
    sub1.set_title("predictions")
    sub2 = sns.countplot(labels,ax=ax[1])
    sub2.set_title("labels")
    plt.savefig("imgs/lstmtesttags.png")
    

# 最终的测试集的表现
with torch.no_grad():
    test_preds, test_labels = [], []
    for batch in test_iter:
        output = model(batch.sent[0], batch.sent[1]).view(-1, OUT_SIZE)
        preds = torch.argmax(output, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(batch.label.view(-1).cpu().numpy())
    test_preds, test_labels = align_predictions(test_preds, test_labels, id2labels)
    cout_tags(test_preds, test_labels)
    print(classification_report(test_labels, test_preds))

sns.lineplot(range(len(loss_list)),loss_list)
plt.savefig('imgs/train_loss_'+ str(BATCH_SZIE) + '.png')