from mimetypes import init
import sys
from numpy import dtype
from tqdm import tqdm
sys.path.append("..")
from transformers import BertTokenizerFast, BertForTokenClassification, AutoTokenizer
from Advtrans.models.bert import BertNERModel
from Advtrans.preprocess import preprocess
from torchtext.legacy.data import Field, Example, Dataset, BucketIterator, LabelField
import torch
import torch.nn as nn
from sklearn.metrics import classification_report


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

cls_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
sep_token = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
pad_token = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
unk_token = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
print(cls_token, sep_token, pad_token, unk_token)
# cls_token = tokenizer.cls_token
# sep_token = tokenizer.sep_token
# pad_token = tokenizer.pad_token
# unk_token = tokenizer.unk_token

# rebuild build_dataset
def build_dataset(tokenizer, train_path=None, dev_path=None, test_path=None):
    """
    train_path: 训练集路径
    valid_path: 验证集路径
    test_path: 测试集路径
    """
    # build fields
    SENT = Field(
        sequential=True,
        batch_first=True,
        # tokenize=tokenizer.tokenize,
        # preprocessing=tokenizer.convert_tokens_to_ids,
        # tokenize=tokenizer.encode,
        use_vocab=False,
        # init_token=cls_token,
        # eos_token=sep_token,
        pad_token=pad_token,
        unk_token=unk_token,
        )
    LABEL = Field(batch_first=True, sequential=True, tokenize=str.split, is_target=True, unk_token=None)
    fields=[('sent', SENT), ('label', LABEL)]

    meger_dataset = {}
    filepaths = {
        'train': train_path,
        'dev': dev_path,
        'test': test_path
    }
    for name, filepath in filepaths.items():
        if filepath is None:
            continue
        sents, labels = preprocess(filepath)
        examples = []
        for sent, label in zip(sents, labels):
            label = label.split()
            sent = sent.split()
            tokenized_input = tokenizer(sent, is_split_into_words=True)
            word_ids = tokenized_input.word_ids()
            # print(word_ids)
            aligned_labels = []
            previous_word_id = None
            for word_id in word_ids:   
                if word_id is None or word_id == previous_word_id:
                    aligned_labels.append('<pad>')
                elif word_id != previous_word_id:
                    aligned_labels.append(label[word_id])
                previous_word_id = word_id

            example = Example.fromlist([tokenized_input.input_ids, aligned_labels], fields)
            # print(example.sent, example.label)
            examples.append(example)
        dataset = Dataset(examples, fields)
        meger_dataset[name] = dataset
    # SENT.build_vocab(meger_dataset['train'], meger_dataset['dev'], meger_dataset['test'])
    LABEL.build_vocab(meger_dataset['train'])
    return meger_dataset
        
lang_dataset = build_dataset( tokenizer, 'data/en/train.txt', 'data/en/dev.txt', 'data/en/test.txt')
print(vars(lang_dataset['train'].examples[0]))
# label vocab
print(lang_dataset['train'].fields['label'].vocab.stoi)

train_iter = BucketIterator(
    lang_dataset['train'],
    batch_size=32,
    sort_key=lambda x: len(x.sent),
    sort_within_batch=True,
    shuffle=True,
)

dev_iter = BucketIterator(
    lang_dataset['dev'],
    batch_size=32,
    sort_key=lambda x: len(x.sent),
    sort_within_batch=True,
    shuffle=True,
)
test_iter = BucketIterator(lang_dataset['test'], batch_size=32, sort_key=lambda x: len(x.sent))

# print(lang_dataset['train'].fields['label'].vocab.stoi)
# print(lang_dataset['train'].fields['label'].vocab.itos)

# load model
num_labels = len(lang_dataset['train'].fields['label'].vocab)
model = BertNERModel(model_name, num_labels)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=0)
EPOCHES = 10
for epoch in range(EPOCHES):
    for step, batch in tqdm(enumerate(train_iter)):
        input_ids = batch.sent
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = (input_ids != pad_token).to(dtype=torch.long)
        labels = batch.label.view(-1)
        outputs = model(input_ids, token_type_ids, attention_mask).view(-1, num_labels)
        # print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], loss:{:.4f}".format(epoch+1, EPOCHES, step, len(train_iter),loss.item()))
    

def align_predictions(predictions, golds, id2labels):
    """
    predictions: [Batch*Length Class]
    labels:[Batch*Length]
    """
    preds_list, labels_list = [], []
    for i in range(len(golds)):
        if golds[i] > 1:
            labels_list.append(id2labels[golds[i]])
            preds_list.append(id2labels[predictions[i]])
    return preds_list, labels_list

# evaluate
def evaluate(model, data_iter, criterion):
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for batch in test_iter:
            input_ids = batch.sent
            token_type_ids = torch.zeros_like(input_ids)
            attention_mask = (input_ids != pad_token).to(dtype=torch.long)
            labels = batch.label.view(-1)
            outputs = model(input_ids, token_type_ids, attention_mask).view(-1, num_labels)
            preds = torch.argmax(outputs, dim=1)
            preds, labels = align_predictions(preds, labels, lang_dataset['train'].fields['label'].vocab.itos)
            preds_list.extend(preds)
            labels_list.extend(labels)
    print(classification_report(labels_list, preds_list))