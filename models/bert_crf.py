import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class BertCRFNERModel(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertCRFNERModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, return_dict=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tags=None, mask=None):
        pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output.last_hidden_state)
        logits = self.classifier(pooled_output)

        # 训练
        if tags is not None:
            loss = -self.crf(logits, tags, mask)
            return loss
        else: # 测试
            logits = self.crf.decode(logits, mask)
            return logits
        