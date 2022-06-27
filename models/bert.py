import torch
import torch.nn as nn
from transformers import BertModel

class BertNERModel(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertNERModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, return_dict=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output.last_hidden_state)
        logits = self.classifier(pooled_output)
        return logits
