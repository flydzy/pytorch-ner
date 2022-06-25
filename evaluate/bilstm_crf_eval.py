import sys
sys.path.append(r'..')

import torch
from Advtrans.preprocess import build_dataset
from torchtext.legacy.data import BucketIterator
from Advtrans.models.bilstm_crf import LSTM_CRF
from torch.optim import Adam
from torch import nn
from sklearn.metrics import classification_report, precision_recall_fscore_support



# BASIC PARAMETERS
BATCH_SIZE = 32



en_dataset = build_dataset(train_path='data/en/eng.train',valid_path='data/en/eng.testa',test_path='data/en/eng.testb')
train_iter = BucketIterator(en_dataset['train'], batch_size=BATCH_SIZE,shuffle=True ,sort_key=lambda x: len(x.sent), sort_within_batch=True, repeat=False)
valid_iter = BucketIterator(en_dataset['valid'], batch_size=BATCH_SIZE, shuffle=True ,sort_key=lambda x: len(x.sent), sort_within_batch=True, repeat=False)
test_iter = BucketIterator(en_dataset['test'], batch_size=BATCH_SIZE, sort_key=lambda x: len(x.sent), sort_within_batch=True, repeat=False)

id2labels = en_dataset['train'].fields['label'].vocab.itos
label2id2 = en_dataset['train'].fields['label'].vocab.stoi


# model parameters
VOCAB_SZIE = en_dataset['train'].fields['sent'].vocab.vectors.size(0)
EMBEDDING_SIZE = en_dataset['train'].fields['sent'].vocab.vectors.size(1)
HIDDEN_SIZE = 100
OUT_SIZE = len(en_dataset['train'].fields['label'].vocab.itos)
IS_BIDIRECTIONAL = True
EMBEDDING = en_dataset['train'].fields['sent'].vocab.vectors
EPOCHES = 10

model = LSTM_CRF(VOCAB_SZIE, EMBEDDING_SIZE, HIDDEN_SIZE, OUT_SIZE, IS_BIDIRECTIONAL, EMBEDDING)
optimizer = Adam(model.parameters())
loss = nn.CrossEntropyLoss(ignore_index=0)


def align_predictions(predictions, golds, id2labels):
    """
    predictions: [Batch*Length Class]
    labels:[Batch*Length]
    """
    preds_list, labels_list = [], []
    for i in range(len(golds)):
        if golds[i] > 0:
            labels_list.append(id2labels[golds[i]])
            preds_list.append(id2labels[predictions[i]])
    return preds_list, labels_list

def get_mask(text, text_len):
    mask = torch.zeros(text.shape, dtype=torch.bool)
    for i in range(len(text_len)):
        mask[i, :text_len[i]] = 1
    return mask

def evaluate(model, data_iter):
    model.eval()
    all_preds, all_labels = [], []
    for batch in data_iter:
        text = batch.sent
        label = batch.label
        mask = get_mask(text[0], text[1])
        preds = model(text[0], text[1], tags = None, mask = mask)
        preds = [label for pred in preds for label in pred ]
        golds = label.masked_select(mask).cpu().numpy()
        preds_list, labels_list = align_predictions(preds, golds, id2labels)
        all_preds.extend(preds_list)
        all_labels.extend(labels_list)
    return all_labels, all_preds

for epoch in range(EPOCHES):
    for step, batch in enumerate(train_iter):
        text = batch.sent
        label = batch.label
        # text[1]表是batch中的每一个句子的原始长度
        # 根据原始长度生成mask
        mask = torch.zeros(text[0].shape, dtype=torch.bool)
        for i in range(len(text[1])):
            mask[i, :text[1][i]] = 1
        loss = model(text[0], text[1], tags = label, mask = mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], loss:{:.4f}".format(epoch+1, EPOCHES, step, len(train_iter),loss.item()))
    
    with torch.no_grad():
        # evaluate on train set
        print("Evaluate on train set:")
        train_labels, train_preds = evaluate(model, train_iter)
        print("precision, recall, f1-score:", precision_recall_fscore_support(train_labels, train_preds, average='weighted'))
        # evaluate on dev set
        print("Evaluate on dev set:")
        dev_labels, dev_preds = evaluate(model, valid_iter)
        print("precision, recall, f1-score:", precision_recall_fscore_support(dev_labels, dev_preds, average='weighted'))
        # evaluate on test set
        print("Evaluate on test set:")
        test_labels, test_preds = evaluate(model, test_iter)
        print("precision, recall, f1-score:", precision_recall_fscore_support(test_labels, test_preds, average='weighted'))