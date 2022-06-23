import  torch
import torch.nn as nn
from Preprocess.preprocess import build_iterator
from models.lstm import LSTM
from sklearn.metrics import classification_report

# load iterator of train, dev, test
train_iter = build_iterator("data/en/eng.train", batch_size=32, device='cpu')
dev_iter = build_iterator("data/en/eng.testa", batch_size=32, device='cpu')
test_iter = build_iterator("data/en/eng.testb", batch_size=32, device='cpu')


# HMM Model evaluation
num_class_hmm = train_iter.dataset.fields["label"].vocab.stoi[]
print("num_class_hmm:", num_class_hmm)
# hmm = 