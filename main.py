from pickletools import optimize
import  torch
import torch.nn as nn
from evaluate import hmm_train_eval, crf_train_eval, lstm_train_eval, lstm_crf_train_eval
from preprocess import build_dataset
import argparse
from torchtext.legacy.data import BucketIterator
import logging
from models.hmm import HMM
from models.crf import CRFModel
from models.bilstm import LSTM
from models.bilstm_crf import LSTM_CRF

parser = argparse.ArgumentParser(description='Evaluate the model')

# data path
parser.add_argument('--train_path', type=str, default='data/en/train.txt', help='path of train data')
parser.add_argument('--dev_path', type=str, default='data/en/dev.txt', help='path of dev data')
parser.add_argument('--test_path', type=str, default='data/en/test.txt', help='path of test data')
parser.add_argument('--lang', type=str, default='en', help='language of data')

# model args
parser.add_argument('--model_path', type=str, default='ckpts/', help='path of model')
parser.add_argument('--epoches', type=int, default=10, help='epoches')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden size')
parser.add_argument('--embedding_size', type=int, default=100, help='embedding size')
parser.add_argument('--out_size', type=int, default=10, help='out size')
parser.add_argument('--bidirectional', type=bool, default=True, help='bidirectional')

# loss function
loss_function = nn.CrossEntropyLoss(ignore_index=0)  # pad was be ignored

# logger args
parser.add_argument('--log_path', type=str, default='logs/', help='path of log')
parser.add_argument('--log_name', type=str, default='log.txt', help='name of log')

# parse the args
args = parser.parse_args()




# data
lang_dataset = build_dataset(train_path=args.train_path, valid_path=args.dev_path, test_path=args.test_path)

origin_dataset = {}
for key in lang_dataset:
    train_sents = [example.sent for example in lang_dataset[key].examples]
    train_labels = [example.label for example in lang_dataset[key].examples]
    origin_dataset[key] = (train_sents, train_labels)

data_iter = {}
data_iter['train'] = BucketIterator(lang_dataset['train'], batch_size=args.batch_size, shuffle=True, sort_key=lambda x: len(x.sent), sort_within_batch=True, repeat=False)
data_iter['dev'] = BucketIterator(lang_dataset['dev'], batch_size=args.batch_size, shuffle=True, sort_key=lambda x: len(x.sent), sort_within_batch=True, repeat=False)
data_iter['test'] = BucketIterator(lang_dataset['test'], batch_size=args.batch_size, sort_key=lambda x: len(x.sent), sort_within_batch=True, repeat=False)


word2ids = lang_dataset['train'].fields['sent'].vocab.stoi
tag2ids = lang_dataset['train'].fields['label'].vocab.stoi
id2labels = lang_dataset['train'].fields['label'].vocab.itos

args.vocab_size = len(word2ids)

# get pretrained embedding
EMBEDDING =  lang_dataset['train'].fields['sent'].vocab.vectors
if EMBEDDING is not None:
    args.embedding_size = EMBEDDING.size(1)

# logger
logger = logging.getLogger(args.log_name)
logger.setLevel(logging.INFO)
# output log to file
handler = logging.FileHandler(args.log_path + args.log_name,mode='a')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# output to console
handler2 = logging.StreamHandler()
handler2.setLevel(logging.INFO)
# set write format
formatter2 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler2.setFormatter(formatter2)

logger.addHandler(handler2)


logger.info("The Language is {}".format(args.lang))
logger.info('Start Processing')


# models
logger.info('*********************Start building models**************************')
hmm = HMM(len(tag2ids), len(word2ids))
crf = CRFModel()
# single directional lstm dont set bidirectional
lstm = LSTM(args.vocab_size, args.embedding_size, args.hidden_size, args.out_size,embedding=EMBEDDING)
bilstm = LSTM(args.vocab_size, args.embedding_size, args.hidden_size, args.out_size, is_bidirectional=args.bidirectional, embedding = EMBEDDING )
bilstm_crf = LSTM_CRF(args.vocab_size, args.embedding_size, args.hidden_size, args.out_size, is_bidirectional=args.bidirectional, embedding = EMBEDDING )
logger.info("*********************Finish building models*********************")


# train and eval
logger.info("*********************Start training models*********************")

logger.info("*********************Start training HMM*********************")
hmm_train_eval(origin_dataset, hmm, word2ids, tag2ids, logger)

logger.info("*********************Start training CRF*********************")
crf_train_eval(origin_dataset, crf, logger)

# loss function
# lstm
logger.info("*********************Start training LSTM*********************")
loss_function_lstm = nn.CrossEntropyLoss(ignore_index=0)  # pad was be ignored
optimizer_for_lstm = torch.optim.Adam(lstm.parameters())
lstm_train_eval(data_iter, lstm, loss_function, optimizer_for_lstm ,args.epoches, args.out_size, id2labels, logger)

# bilstm
logger.info("*********************Start training BiLSTM*********************")
loss_function_bilstm = nn.CrossEntropyLoss(ignore_index=0)  # pad was be ignored
optimizer_for_bilstm = torch.optim.Adam(bilstm.parameters())
lstm_train_eval(data_iter, bilstm, loss_function, optimizer_for_bilstm ,args.epoches, args.out_size, id2labels, logger)

# bilstm_crf
logger.info("*********************Start training BiLSTM_CRF*********************")
optimizer_for_bilstm_crf = torch.optim.Adam(bilstm_crf.parameters())
lstm_crf_train_eval(data_iter, bilstm_crf, optimizer_for_bilstm_crf, args.epoches, id2labels, logger)



# sava models
logger.info("*********************Start saving models*********************")
torch.save(hmm, args.model_path + args.lang + '_hmm.pkl')
torch.save(crf, args.model_path + args.lang + '_crf.pkl')
torch.save(lstm, args.model_path + args.lang + '_lstm.pkl')
torch.save(bilstm, args.model_path + args.lang + '_bilstm.pkl')
torch.save(bilstm_crf, args.model_path + args.lang + '_bilstm_crf.pkl')
