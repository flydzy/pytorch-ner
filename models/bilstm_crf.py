from torch import nn
from torchcrf import CRF
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, out_size, is_bidirectional, embedding):
        super(LSTM_CRF, self).__init__()
        
        # use pre-trained embedding or not
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding)
        
        # is bidirectional or not
        if is_bidirectional:
            self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional = True, num_layers = 1)
            self.dropout = nn.Dropout(p = 0.5)
            self.liner = nn.Linear(2 * hidden_size, out_size)
            # add CRF
            
        else:
            self.lstm = nn.LSTM(embedding_size, hidden_size)
            self.dropout = nn.Dropout(p=0.5)
            self.liner = nn.Linear(hidden_size, out_size)
            # add CRF

        self.crf = CRF(out_size, batch_first=True)
        
    def forward(self, sents_tensor, lengths, tags=None, mask=None):
        emb = self.embedding(sents_tensor)
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.dropout(out)
        out = self.liner(out)
        # add CRF
        if tags is not None:  # training
            if mask is not None:
                loss = -self.crf(out, tags, mask)
            else:
                loss = -self.crf(out, tags)
            return loss
        else:   # test
            if mask is not None:
                scores = self.crf.decode(out, mask)
            else:
                scores = self.crf.decode(out)
            return scores

