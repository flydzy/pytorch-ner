from torchtext.legacy.data import Field, Example, Dataset, BucketIterator, Iterator
from tqdm import tqdm
import torch

SENT = Field(sequential=True, tokenize=str.split, lower=True, batch_first=True, include_lengths=True) 
LABEL = Field(sequential=True, tokenize=str.split, is_target=True, batch_first=True)

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
def build_dataset(filepath):
    """
    param sents: 句子列表
    param labels: 标签列表
    return: torchtext的Example对象
    """
    # 创建Field
    # include_lengths=True 表示最终返回的是一个tuple，第一个是句子，第二个是句子长度
    
    # 创建Example
    sents, labels = preprocess(filepath)
    fields = [('sent', SENT), ('label', LABEL)]
    examples = []
    for sent, label in zip(sents, labels):
        example = Example.fromlist([sent, label], fields)
        examples.append(example)

    # SENT.build_vocab([example.sent for example in examples], vectors='glove.6B.50d')
    # LABEL.build_vocab([example.label for example in examples], sorted=True)
    dataste = Dataset(examples, fields)
    return dataste

# 构建迭代器
def build_iterator(file_path,  batch_size, device):
    """
    param dataset: torchtext的Dataset对象
    param batch_size: batch大小
    return: 迭代器
    """
    sents, labels = preprocess(file_path)
    dataset = build_dataset(sents, labels)

    iterator = BucketIterator(
        dataset,
        batch_size=batch_size,
        device=device,
        repeat=False,
        shuffle=True,
        sort_key=lambda x: len(x.sent),
        sort_within_batch=True,
        )
    return dataset,iterator

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, data_iter = build_iterator('data/en/eng.train', batch_size=4, device=device)
    label2idx = data_iter.dataset.fields['label'].vocab.stoi
    print(label2idx)
    for i, batch in enumerate(data_iter):
        # print(batch.dataset.examples[0].label)
        print(batch)
        print(batch.sent[0])
        print(batch.sent[1])
        print(batch.label[0])
        break

