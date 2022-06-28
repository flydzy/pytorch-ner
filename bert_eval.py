from models.bert import BertNERModel
import torch.nn as nn
from preprocess2 import build_iterators, categories
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch

train_iter = build_iterators(data_path='data/en/train.txt', batch_size=32, shuffle=True)
dev_iter = build_iterators(data_path='data/en/dev.txt', batch_size=32, shuffle=True)
test_iter = build_iterators(data_path='data/en/test.txt', batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_class = 9
model_name = 'bert-base-uncased'
model = BertNERModel(model_name, num_class).to(device)
EPOCHES = 10
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

for epoch in EPOCHES:
    model.train()
    for i, batch in tqdm(enumerate(train_iter), total=len(train_iter)):
        input_ids = batch.input_ids.to(device)
        token_type_ids = batch.token_type_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)
        outputs = model(input_ids, token_type_ids, attention_mask).view(-1, num_class)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], loss:{:.4f}".format(epoch+1, EPOCHES, i, len(train_iter),loss.item()))

def align_predictions(predictions, golds, id2labels):
    """
    predictions: [Batch*Length Class]
    labels:[Batch*Length]
    """
    preds_list, labels_list = [], []
    for i in range(len(golds)):
        if golds[i] != -100:
            labels_list.append(id2labels[golds[i]])
            preds_list.append(id2labels[predictions[i]])
    return preds_list, labels_list

def evaluate(model, data_iter, id2labels):
    model.eval()
    preds_list, labels_list = [], []
    for batch in tqdm(data_iter):
        input_ids = batch.input_ids.to(device)
        token_type_ids = batch.token_type_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        outputs = model(input_ids, token_type_ids, attention_mask).view(-1, num_class)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        golds = batch.labels.view(-1)
        preds, golds = align_predictions(preds, golds, id2labels)
        preds_list.extend(preds)
        labels_list.extend(golds)
    preds, labels = align_predictions(preds, labels, id2labels)
    return classification_report(labels, preds)

# dev evaluation
print(evaluate(model, dev_iter, categories))
# test evaluation
print(evaluate(model, test_iter, categories))
