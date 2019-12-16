# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 17:40
# @Author  : zhoujun

# data config
trainroot = 'Training Set'
testroot = 'Test Set'
output_dir = '/content/drive/My Drive/PSENet_2'
pretrained_path = '/content/drive/My Drive/PSENet_2/PSENet_resnet50.pth'
data_shape = 640

# train config
gpu_id = '0'
workers = 0
start_epoch = 0
epochs = 300

train_batch_size = 4

lr = 0.5e-5
end_lr = 1e-6
lr_gamma = 0.1
lr_decay_step = [100,200]
weight_decay = 5e-4
warm_up_epoch = 6
warm_up_lr = lr * lr_gamma

display_input_images = False
display_output_images = False
display_interval = 10
show_images_interval = 50

pretrained = False
restart_training = False
checkpoint = ''

# net config
backbone = 'resnet50'
Lambda = 0.7
n = 6
m = 0.5
OHEM_ratio = 3
scale = 1
# random seed
seed = 2


def print():
    from pprint import pformat
    tem_d = {}
    for k, v in globals().items():
        if not k.startswith('_') and not callable(v):
            tem_d[k] = v
    return pformat(tem_d)
