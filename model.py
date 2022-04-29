import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers import BertTokenizerFast


class CscModel(nn.Module):
    def __init__(self, config):
        super(CscModel, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(config['model_path'])
        self.ff = nn.Linear(config['hidden_size'], 1)

    def forward(self, input_ids, token_type_ids, attention_mask) -> (torch.Tensor, torch.Tensor):
        bert_encoder = self.bert(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 output_hidden_states=True)
        logits = bert_encoder.logits
        det_output = bert_encoder.hidden_states[-1]
        det_output = self.ff(det_output).squeeze(-1)
        det_output = torch.sigmoid(det_output)

        return det_output, logits


class CscModelCorrector(object):
    def __init__(self, model: CscModel, tokenizer: BertTokenizerFast, model_config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config

    def correct(self, text):
        encoded_text = self.tokenizer(text, return_offsets_mapping=False,
                                      max_length=self.model_config['max_len'], truncation=True,
                                      padding='max_length', return_tensors='pt',
                                      return_attention_mask=True, return_token_type_ids=True)
        with torch.no_grad():
            det_output, logits = self.model(**encoded_text)
        token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
        decode_tokens = self.tokenizer.decode(token_ids[0], skip_special_tokens=False)
        corrected_text = decode_tokens[1:len(text)+1]
        return corrected_text, det_output[0][1:len(text)+1]
