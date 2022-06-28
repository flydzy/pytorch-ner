import sys
sys.path.append('../')
from Advtrans.preprocess2 import build_iterators, categories
from Advtrans.models.bert_crf import BertCRFNERModel
from tqdm import tqdm, tgrange
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from Advtrans.utils import align_predictions

# model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'bert-base-uncased'
num_classes = 10
EPOCHES = 10

train_iter = build_iterators('data/en/train.txt', batch_size=32, shuffle=True)
dev_iter = build_iterators('data/en/dev.txt', batch_size=32, shuffle=True)
test_iter = build_iterators('data/en/test.txt', batch_size=32, shuffle=False)

model = BertCRFNERModel(model_name, num_classes).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss(ignore_index=-100)

# for epoch in range(EPOCHES):
#     for step, batch in tqdm(enumerate(train_iter), total=len(train_iter), leave=False):
#         input_ids = batch[0].input_ids.to(device)
#         token_type_ids = batch[0].token_type_ids.to(device)
#         attention_mask = batch[0].attention_mask.to(device)
#         mask = attention_mask.clone().to(torch.bool)
#         labels = batch[1].to(device)
#         loss = model(input_ids, token_type_ids, attention_mask, labels, mask)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if step % 100 == 0:
#             print("Epoch [{}/{}], Step [{}/{}], loss:{:.4f}".format(epoch+1, EPOCHES, step, len(train_iter),loss.item()))

# torch.save(model, "ckpts/bert_crf.pt")
torch.load("ckpts/bert_crf.pt")

def evaluate(model, data_iter, id2labels):
    model.eval()
    preds_list, labels_list = [], []
    for batch in tqdm(data_iter):
        input_ids = batch[0].input_ids.to(device)
        token_type_ids = batch[0].token_type_ids.to(device)
        attention_mask = batch[0].attention_mask.to(device)
        outputs = model(input_ids, token_type_ids, attention_mask)
        golds = batch[1].view(-1).cpu().numpy()
        preds = [label for pred in outputs for label in pred]
        preds, golds = align_predictions(preds, golds, id2labels)
        preds_list.extend(preds)
        labels_list.extend(golds)
        # print(preds_list, labels_list)
        # break
    # print(preds_list, labels_list)
    # final_preds, final_labels = align_predictions(preds_list, labels_list, id2labels)
    # print(preds_list, labels_list)
    # print(final_labels)
    return classification_report(labels_list, preds_list)

# evaluate on dev set
print(evaluate(model, dev_iter, categories))
# evaluate on test set
print(evaluate(model, test_iter, categories))