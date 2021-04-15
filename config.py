class Config:
    data_path = "figures"
    model_path_train = ""
    model_path_test = "figures/checkpoint/model_20.ckpt" 
    output_path = "results"

    img_size = 423
    adjust_size = 423
    train_size = 423
    img_channel = 3
    conv_channel_base = 64

    learning_rate = 0.0002
    beta1 = 0.5
    max_epoch = 20
    L1_lambda = 100
    save_per_epoch=1
