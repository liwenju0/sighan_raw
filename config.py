
class cfg:
    hidden_size = 768
    model_path = "hfl/chinese-macbert-base"
    max_len = 128
    vocab_size = 21128
    lr = 5e-5
    clip_norm = 0.25
    weight_decay = 0.01
    batch_size = 32
    max_epochs = 10
    loss_weight = 0.3
    train = "train13.json"
    test = "test13.json"
    base_url = "/Users/milter/Downloads/sighan_raw/pair_data/simplified/"
    # base_url="/root/liwenju/sighan_raw/pair_data/simplified/",
