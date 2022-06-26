from torchtext.legacy.data import Field, Example, Dataset, BucketIterator, Iterator
from tqdm import tqdm
import torch

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
                    # for i in range(len(temp_label)):
                    #     # 判断是不是实体
                    #     if temp_label[i] == 'O':
                    #         continue
                    #     else:
                    #         # 判断是不是第一个
                    #         if i == 0:        
                    #             temp_label[i] = 'B-' + temp_label[i].split('-')[-1]
                    #         # 判断是不是最后一个
                    #         elif i == len(temp_label) - 1 and temp_label[i-1].split('-')[-1] != temp_label[i].split('-')[-1]:
                    #             temp_label[i] = 'B-' + temp_label[i].split('-')[-1]
                    #         # 中间的实体 O B-实体 I-实体这种形式的
                    #         elif temp_label[i-1].split('-')[-1] != temp_label[i].split('-')[-1]:
                    #             temp_label[i] = 'B-' + temp_label[i].split('-')[-1]
                    sents.append(" ".join(temp_sent))
                    labels.append(" ".join(temp_label))
                    temp_sent = []
                    temp_label = []
            else:
                temp_sent.append(line.split()[0])
                temp_label.append(line.split()[-1])

    return sents, labels                

# 代入到torchtext中
def build_dataset(train_path, valid_path=None, test_path=None):
    """
    train_path: 训练集路径
    valid_path: 验证集路径
    test_path: 测试集路径
    """

    # build fields
    SENT = Field(sequential=True, tokenize=str.split, lower=True, batch_first=True, include_lengths=True) 
    LABEL = Field(sequential=True, tokenize=str.split, unk_token=None ,is_target=True, batch_first=True)

    meger_dataste = {}
    filepaths = {
        'train': train_path,
        'dev': valid_path,
        'test': test_path
    }
    
    for name,filepath in filepaths.items():
        if filepath is None:
            continue
        sents, labels = preprocess(filepath)
        fields = [('sent', SENT), ('label', LABEL)]
        examples = []
        for sent, label in zip(sents, labels):
            example = Example.fromlist([sent, label], fields)
            examples.append(example)
        dataste = Dataset(examples, fields)
        meger_dataste[name] = dataste
    # build vocab
    SENT.build_vocab(meger_dataste['train'], vectors='glove.6B.300d')
    LABEL.build_vocab(meger_dataste['train'])
    return meger_dataste


if __name__ == '__main__':
    pass