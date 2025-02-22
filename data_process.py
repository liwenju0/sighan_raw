# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description:
"""
import os
import operator
import json
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤', '\t', '֍', '玕', '']
def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if i >= len(corrected_text):
            continue
        if ori_char in unk_tokens:
            # deal with unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details

def split_text_by_maxlen(text, maxlen=512):
    """
    文本切分为句子，以句子maxlen切分
    :param text: str
    :param maxlen: int, 最大长度
    :return: list, (sentence, idx)
    """
    result = []
    for i in range(0, len(text), maxlen):
        result.append((text[i:i + maxlen], i))
    return result

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        ori_texts, cor_texts, wrong_idss = zip(*data)
        encoded_texts = [self.tokenizer.tokenize(t) for t in ori_texts]
        max_len = max([len(t) for t in encoded_texts]) + 2
        det_labels = torch.zeros(len(ori_texts), max_len).long()
        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            for idx in wrong_ids:
                margins = []
                for word in encoded_text[:idx]:
                    if word == '[UNK]':
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while (abs(move) < margin) or (idx + move >= len(encoded_text)) \
                        or encoded_text[idx + move].startswith('##'):
                    move -= 1
                det_labels[i, idx + move + 1] = 1
        return ori_texts, cor_texts, det_labels


class CscDataset(Dataset):
    def __init__(self, file_path):
        self.data = json.load(open(file_path, 'r', encoding='utf-8'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']


def make_loaders(collate_fn, train_path='', valid_path='', test_path='',
                 batch_size=32, num_workers=0):
    train_loader = None
    if train_path and os.path.exists(train_path):
        train_loader = DataLoader(CscDataset(train_path),
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    valid_loader = None
    if valid_path and os.path.exists(valid_path):
        valid_loader = DataLoader(CscDataset(valid_path),
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    test_loader = None
    if test_path and os.path.exists(test_path):
        test_loader = DataLoader(CscDataset(test_path),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader
