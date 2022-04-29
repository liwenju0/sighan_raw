import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from config import model_config


class CscDataset(Dataset):
    def __init__(self, config, name="15", key_name="train"):
        self.max_len = config['max_len']
        self.pretrain_model = config['model_path']
        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrain_model, do_lower_case=True)
        self.error = config['base_url'] + "{}{}_error.txt".format(key_name, name)
        self.correct = config['base_url']+"{}{}_correct.txt".format(key_name, name)
        self.data_pair = []
        for e, c in zip(open(self.error), open(self.correct)):
            assert len(e.strip()) == len(c.strip())
            self.data_pair.append((e.strip(), c.strip()))

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, index):
        error_text, correct_text = self.data_pair[index]

        err_ids = []
        for ind, (e, c) in enumerate(zip(error_text, correct_text)):
            if e != c:
                err_ids.append(ind)

        econtext = self.tokenizer(error_text, return_offsets_mapping=True,
                                  max_length=self.max_len, truncation=True,
                                  padding='max_length', return_tensors='pt',
                                  return_attention_mask=True, return_token_type_ids=True)
        cor_labels = self.tokenizer(correct_text, return_offsets_mapping=True,
                                    max_length=self.max_len, truncation=True,
                                    padding='max_length', return_tensors='pt',
                                    return_attention_mask=True, return_token_type_ids=True)['input_ids'][0]
        det_labels = []
        for tok_id, (start, end) in enumerate(econtext['offset_mapping'][0]):
            if start == end:
                det_labels.append(-100)
            else:
                if start in err_ids:
                    det_labels.append(1)
                else:
                    det_labels.append(0)

        cor_labels[cor_labels == 0] = -100

        return econtext['input_ids'][0], econtext['token_type_ids'][0], econtext['attention_mask'][
            0], cor_labels, torch.LongTensor(det_labels)


if __name__ == '__main__':
    cs = CscDataset(model_config)
    print(len(cs.tokenizer.get_vocab()))
    c = cs[0]
    print("d")
