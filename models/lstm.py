import torch
from torch import nn
import torch.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size, embedding=None):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(LSTM, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding)
        self.lstm = nn.LSTM(emb_size, hidden_size, bidirectional=True, num_layers=1)

        # 训练集效果很好，但是验证集效果差，加入dropout层
        self.dropout = nn.Dropout(p = 0.5)

        self.liner = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)
        packed = pack_padded_sequence(emb, lengths, batch_first=True)  # 将填充的部分pack，不参与计算
        out,_ = self.lstm(packed)
        out,_ = pad_packed_sequence(out, batch_first=True)  # 将填充的部分pad回tensor，进行计算
        out = self.dropout(out)
        scores = self.liner(out)
        return scores
    
    def test(self, sents_tensor, lengths, _):
        """
        第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口
        """
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids
