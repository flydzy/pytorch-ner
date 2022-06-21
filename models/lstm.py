from struct import pack
from turtle import forward
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first = True)
        self.liner = nn.Linear(hidden_size, out_size)

    def forward(self, sents_tensor):
        emb = self.embedding(sents_tensor)
        # packed = pack_padded_sequence(emb, lengths, batch_first=True)
        out,(hn, cn) = self.lstm(emb)
        # out,_ = pad_packed_sequence(out, batch_first=True)
        scores = self.liner(out)
        return scores
    
    def test(self, sents_tensor, lengths, _):
        """
        第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口
        """
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids
