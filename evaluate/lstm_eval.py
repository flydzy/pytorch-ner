import sys
import torch
from zmq import device
sys.path.append(r'..')
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from Advtrans.models.lstm import LSTM
from Advtrans.preprocess.preprocess import build_iterator
import torch.optim as optim 
import numpy as np
from seqeval.metrics import f1_score, classification_report


train_iter = build_iterator('data/en/eng.train', batch_size=32, device='cpu')
valid_iter = build_iterator('data/en/eng.testa', batch_size=32, device='cpu')
test_iter = build_iterator('data/en/eng.testb', batch_size=32, device='cpu')


# vocab_size 
word2idx = train_iter.dataset.fields['sent'].vocab.stoi
label2idx = train_iter.dataset.fields['label'].vocab.stoi
vocab_size = len(word2idx)

hidden_size = 300
out_size = len(label2idx)
epoches = 10
print(vocab_size, out_size,label2idx)
embedding = train_iter.dataset.fields['sent'].vocab.vectors
emb_size = embedding.shape[1]

model = LSTM(vocab_size, emb_size, hidden_size, out_size,embedding)
optimizer = optim.Adam(model.parameters(),lr=0.01)
loss_fuc = nn.CrossEntropyLoss(ignore_index=1)


def align_predictions(predictions, labels, id2labels):
    """
    predictions: [Batch Length Class]
    labels:[Batch Class]
    """
    preds = torch.argmax(predictions, dim=2)  # [B L]
    batch_size, seq_length = preds.shape
    labels_list, preds_list = [], []
    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_length):
            if labels[batch_idx, seq_idx] != 1:
                example_labels.append(id2labels[labels[batch_idx][seq_idx]])
                example_preds.append(id2labels[preds[batch_idx][seq_idx]])
        labels_list.append(example_labels)
        preds_list.append(example_preds)
    
    return preds_list, labels_list

loss_list = []
for epoch in range(epoches):
    for step,batch in enumerate(train_iter):
        labels = batch.label
        labels = labels.view(labels.shape[0]*labels.shape[1])
        output = model(batch.sent[0], batch.sent[1]).view(-1, out_size) # [B L N]
        loss = loss_fuc(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], loss:{:.4f}".format(epoch+1,epoches, step,len(train_iter),loss.item()))
    loss_list.append(loss.item())
    # 验证集评估准确率
    id2labels = train_iter.dataset.fields['label'].vocab.itos
    preds_list,labels_list = [],[]
    with torch.no_grad():
        for batch in valid_iter:
            labels = batch.label
            preds = model(batch.sent[0], batch.sent[1])
            p, l = align_predictions(preds, labels, id2labels)
            # print(p, l)
            preds_list.extend(p)
            labels_list.extend(l)

    print(classification_report(labels_list, preds_list))