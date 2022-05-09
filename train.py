from model import CscModel
from transformers import AdamW
from transformers import BertTokenizer
from config import cfg
from data_process import DataCollator, make_loaders

tokenizer = BertTokenizer.from_pretrained(cfg.model_path)
collator = DataCollator(tokenizer=tokenizer)
# 加载数据
train_loader, valid_loader, test_loader = make_loaders(collator, train_path=cfg.base_url + cfg.train,
                                                       valid_path=cfg.base_url + cfg.test,
                                                       test_path=cfg.base_url + cfg.test,
                                                       batch_size=cfg.batch_size, num_workers=0)

model = CscModel(tokenizer)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=cfg.lr)

epoches = 20
for i in range(epoches):
    for step, (ori_texts, cor_texts, det_labels) in enumerate(train_loader):
        det_loss, cor_loss, det_pred, cor_pred, cons_loss = model(ori_texts, cor_labels=cor_texts, det_labels=det_labels)
        loss = 0.2*det_loss + 0.6*cor_loss + 0.2*cons_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            loss = loss.item()
            print(f"loss: {loss:>5f}")
            model.evaluate(cfg.base_url+cfg.test)
