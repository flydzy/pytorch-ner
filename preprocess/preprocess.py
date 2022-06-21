from posixpath import split
from numpy import pad
from requests import delete
import torch
from torchtext.legacy.data import Field, Example, Dataset, BucketIterator, Iterator
from tqdm import tqdm

# 对现有数据集做一个转换
def preprocess(data_path):
    """
    param data_path: 数据集路径
    return: 处理好的数据
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sents  = []
        labels = []
        temp_sent = []
        temp_label = []
        for line in tqdm(lines):
            line = line.strip()
            if line == "":
                if len(temp_sent) != 0:
                    for i in range(len(temp_label)):
                        # 判断是不是实体
                        if temp_label[i] == 'O':
                            continue
                        else:
                            # 判断是不是第一个
                            if i == 0:        
                                temp_label[i] = 'B-' + temp_label[i].split('-')[-1]
                            # 判断是不是最后一个
                            elif i == len(temp_label) - 1 and temp_label[i-1].split('-')[-1] != temp_label[i].split('-')[-1]:
                                temp_label[i] = 'B-' + temp_label[i].split('-')[-1]
                            # 中间的实体 O B-实体 I-实体这种形式的
                            elif temp_label[i-1].split('-')[-1] != temp_label[i].split('-')[-1]:
                                temp_label[i] = 'B-' + temp_label[i].split('-')[-1]
                    sents.append(" ".join(temp_sent))
                    labels.append(" ".join(temp_label))
                    temp_sent = []
                    temp_label = []
            else:
                temp_sent.append(line.split()[0])
                temp_label.append(line.split()[-1])

    return sents, labels                

# 代入到torchtext中
def build_dataset(sents, labels):
    """
    param sents: 句子列表
    param labels: 标签列表
    return: torchtext的Example对象
    """
    # 创建Field
    # 保持原有的大小写

    SENT = Field(sequential=True, tokenize=str.split ,lower=False, batch_first=True, fix_length=60)
    LABEL = Field(sequential=True, tokenize=str.split ,pad_token='O',unk_token=None,batch_first=True, fix_length=60)
    # 创建Example
    fields = [('sent', SENT), ('label', LABEL)]
    examples = []
    for sent, label in zip(sents, labels):
        example = Example.fromlist([sent, label], fields)
        examples.append(example)

    SENT.build_vocab([example.sent for example in examples], vectors='glove.6B.50d')
    # 自定义词表
    # LABEL.build_vocab([example.label for example in examples])
    dataste = Dataset(examples, fields)
    return dataste

# 构建迭代器
def build_iterator(file_path,  batch_size):
    """
    param dataset: torchtext的Dataset对象
    param batch_size: batch大小
    return: 迭代器
    """
    sents, labels = preprocess(file_path)
    dataset = build_dataset(sents, labels)

    iterator = Iterator(
        dataset, 
        batch_size=batch_size, 
        device="cpu", 
        repeat=False, 
        sort_key=lambda x: len(x.sent), 
        sort_within_batch=False,
        )
    return iterator

if __name__ == '__main__':
    data_iter = build_iterator('test.txt', batch_size=8)
    for i, batch in enumerate(data_iter):
        print(batch.sent[0])
        print(batch.label[0])
        break

