import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

categories = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6,
    'B-MISC': 7,
    'I-MISC': 8,
    0: 'O',
    1: 'B-PER',
    2: 'I-PER',
    3: 'B-ORG',
    4: 'I-ORG',
    5: 'B-LOC',
    6: 'I-LOC',
    7: 'B-MISC',
    8: 'I-MISC'
}



def load_sentence_tags(data_path):
    """
    Load sentence and tags from file.
    """
    data = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        temp_sent, temp_label = [],[]
        for line in lines:
            line = line.strip()
            if not line and temp_sent is not None:
                data.append((temp_sent, temp_label))
                temp_sent = []
                temp_label = []
            else:
                sent, label = line.split()
                temp_sent.append(sent)
                temp_label.append(label)
    return data

class ConllDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.dataset[idx]



# 处理数据的回调函数
def collate_fn(examples):
    """
    Convert a list of examples to features that can be fed to the model.
    """
    sents, all_labels = [], []
    for sent, ner_labels in examples:
        sents.append(sent)
        all_labels.append([categories[label] for label in ner_labels])
    tokenized_input = tokenizer(sents, 
                                truncation=True, 
                                padding=True, 
                                return_offsets_mapping=True, 
                                is_split_into_words=True,
                                max_length=128, 
                                return_tensors='pt')
    targets = []
    for i, labels in enumerate(all_labels):
        aligned_labels = []
        for word_idx in tokenized_input.word_ids(batch_index=i):
            if word_idx is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(labels[word_idx])
        targets.append(aligned_labels)
    return tokenized_input, targets

if __name__ == '__main__':
    data_path = 'data/en/train.txt'
    train_data = load_sentence_tags(data_path)
    dataset = ConllDataset(train_data)
    print(dataset[0])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    for batch in data_loader:
        print(batch)
        break

