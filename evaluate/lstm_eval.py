from cProfile import label
from pickletools import optimize
import sys
from this import s
from turtle import shape
from numpy import argmax

import torch
sys.path.append(r'..')
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from Advtrans.models.lstm import LSTM
from Advtrans.preprocess.preprocess import build_iterator
import torch.optim as optim 
import time
train_iter = build_iterator('data/en/eng.train', batch_size=32)
valid_iter = build_iterator('data/en/eng.testa', batch_size=32)
test_iter = build_iterator('data/en/eng.testb', batch_size=16)
# vocab_size 
word2idx = train_iter.dataset.fields['sent'].vocab.stoi
label2idx = train_iter.dataset.fields['label'].vocab.stoi
vocab_size = len(word2idx)
emb_size = 60
hidden_size = 60
out_size = len(label2idx)
epoches = 5
print(vocab_size, out_size,label2idx)


model = LSTM(vocab_size, emb_size, hidden_size, out_size)
optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=0.005)
loss_fuc = nn.CrossEntropyLoss()

def cal_loss(logits, targets, label2id):
    """计算损失
    参数:
        logits: [B, L, out_size]
        targets: [B, L]
        lengths: [B]
    """
    PAD = label2id.get('<pad>')
    assert PAD is not None

    mask = (targets != PAD)  # [B, L]
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)

    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)

    return loss


loss_list = []
for epoch in range(epoches):
    for step,batch in enumerate(train_iter):
        labels = batch.label  # [B L]
        # PAD = label2idx.get('<pad>')
        # mask = (labels != PAD)
        # labels = 
        output = model(batch.sent) # [B L O]
        loss = cal_loss(output, labels, label2idx)
        # loss = loss_fuc(output.float(), batch.label.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], loss:{:.4f}".format(epoch+1,epoches, step,len(train_iter),loss.item()))
    loss_list.append(loss.item())
    # 验证集评估准确率
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_valid in valid_iter:
            labels = batch_valid.label
            PAD = label2idx.get('<pad>')
            UNK = label2idx.get('<unk>')
            mask1 = torch.tensor(labels != PAD)
            mask2 = torch.tensor(labels != UNK)
            maks3 = torch.tensor(label != 2)
            mask = ((mask1 == mask2)== maks3)
            labels = torch.masked_select(labels, mask)
            output = model(batch_valid.sent)
            output = torch.argmax(output, dim=2)
            output = torch.masked_select(output, mask)
            correct += output.eq(labels).sum().item()
            total += len(labels)
        print("The Accuracy of Valid:{}".format(correct/total))



sns.lineplot(x=range(len(loss_list)),y=loss_list, markers=True)
plt.savefig("lstm_loss.png")
# 测试集
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for batch_test in test_iter:
#         labels = batch_test.label
#         PAD = label2idx.get('<pad>')
#         UNK = label2idx.get('<unk>')
#         mask1 = torch.tensor(labels != PAD)
#         mask2 = torch.tensor(labels != UNK)
#         mask = mask1 == mask2
#         labels = torch.masked_select(labels, mask)
#         output = model(batch_test.sent)
#         output = torch.argmax(output, dim=2)
#         output = torch.masked_select(output, mask)
#         correct += output.eq(labels).sum().item()
#         total += len(labels)
#     print("The Accuracy of Test:{}".format(correct/total))

# case
batch = next(iter(test_iter))
# labels = test_iter.dataset.examples[0].label
sent = batch.sent[5]
sent = sent.unsqueeze(0)
label = batch.label[5]
output = model(sent)
output = torch.argmax(output,dim=-1)

id2label = train_iter.dataset.fields['label'].vocab.itos
print(id2label)
predict = []
target = []
print(output.squeeze_(0))
print(label)
for pre,tar in zip(output.squeeze_(0), label):
    predict.extend([id2label[pre]])
    target.extend([id2label[tar]])
print(predict)
print(target)