import torch

from model import CscModel
from data_process import CscDataset
from config import model_config
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertTokenizerFast
from model import CscModelCorrector
import torch.nn.functional as F

train_data = CscDataset(config=model_config, name="13", key_name="train")
test_data = CscDataset(config=model_config, name="13", key_name="test")

train_loader = DataLoader(train_data, shuffle=True, batch_size=6)
test_loader = DataLoader(test_data, shuffle=False, batch_size=6)

model = CscModel(config=model_config)
model_corrector = CscModelCorrector(model, train_data.tokenizer)
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=model_config['lr'])


def evaluate(predict_func, error="test13_error.txt",
             correct="test13_correct.txt"):
    error = model_config['base_url'] + error
    correct = model_config['base_url'] + correct
    """
    句级评估结果，设定需要纠错为正样本，无需纠错为负样本
    X需要纠正且纠正正确
    Y纠正的句子数
    Z总共需要纠正的句子数
    """

    X, Y, Z = 1e-10, 1e-10, 1e-10
    Xc =  1e-10

    for e, c in zip(open(error), open(correct)):
        e = e.strip()
        c = c.strip()
        corrected_text, det_ouput = predict_func(e)

        if e != c:
            Z += 1
            if corrected_text == c:
                X += 1
            flag = True
            for i in range(len(e)):
                if not (e[i] != c[i] and det_ouput[i] > 0.5):
                    flag = False
            if flag:
                Xc += 1
        if corrected_text != e:
            Y += 1
    print("correct level: precision:{} recall:{},f1:{}".format(X/Y, X/Z, 2*X/(Y+Z)))
    print("detect level: precision:{} recall:{},f1:{}".format(Xc/Y, Xc/Z, 2*Xc/(Y+Z)))











epoches = 10
for i in range(epoches):

    size = len(train_loader.dataset)
    for step, (input_ids, token_type_ids, attention_mask, cor_labels, det_labels) in enumerate(train_loader):
        det_output, logits = model(input_ids, token_type_ids, attention_mask)
        det_loss = torch.nn.CrossEntropyLoss()(det_output, det_labels.float())
        cor_loss = torch.nn.CrossEntropyLoss()(logits.view(-1, model_config['vocab_size']),
                                                              cor_labels.view(-1,))
        loss = det_loss + cor_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            loss, current = loss.item(), step * len(input_ids)
            print(f"loss: {loss:>5f}  [{current:>5d}/{size:>5d}]")
        evaluate(model_corrector.correct)
