# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import cv2
import os
import config

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

import shutil
import glob
import time
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torch.utils.data as Data
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from dataset.data_utils import MyDataset
from models import PSENet
from models.loss import PSELoss
from utils.utils import load_checkpoint, save_checkpoint, setup_logger
from pse import decode as pse_decode
from cal_recall import cal_recall_precison_f1


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# learning rate的warming up操作
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < config.warm_up_epoch:
        lr = 1e-6 + (config.lr - 1e-6) * epoch / (config.warm_up_epoch)
    else:
        lr = config.lr * (config.lr_gamma ** (epoch / config.lr_decay_step[0]))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_epoch(net, optimizer, scheduler, train_loader, device, criterion, epoch, all_step, writer, logger):
    net.train()
    train_loss = 0.
    start = time.time()
    scheduler.step()
    # lr = adjust_learning_rate(optimizer, epoch)
    lr = scheduler.get_lr()[0]
    for i, (images, labels, training_mask) in enumerate(train_loader):
        cur_batch = images.size()[0]
        images, labels, training_mask = images.to(device), labels.to(device), training_mask.to(device)
        # Forward
        y1 = net(images)
        loss_c, loss_s, loss = criterion(y1, labels, training_mask)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loss_c = loss_c.item()
        loss_s = loss_s.item()
        loss = loss.item()
        cur_step = epoch * all_step + i
        writer.add_scalar(tag='Train/loss_c', scalar_value=loss_c, global_step=cur_step)
        writer.add_scalar(tag='Train/loss_s', scalar_value=loss_s, global_step=cur_step)
        writer.add_scalar(tag='Train/loss', scalar_value=loss, global_step=cur_step)
        writer.add_scalar(tag='Train/lr', scalar_value=lr, global_step=cur_step)

        if i % config.display_interval == 0:
            batch_time = time.time() - start
            logger.info(
                '[{}/{}], [{}/{}], step: {}, {:.3f} samples/sec, batch_loss: {:.4f}, batch_loss_c: {:.4f}, batch_loss_s: {:.4f}, time:{:.4f}, lr:{}'.format(
                    epoch, config.epochs, i, all_step, cur_step, config.display_interval * cur_batch / batch_time,
                    loss, loss_c, loss_s, batch_time, lr))
            start = time.time()
            
    writer.add_scalar(tag='Train_epoch/loss', scalar_value=train_loss / all_step, global_step=epoch)
    return train_loss / all_step, lr


def main():
    if config.output_dir is None:
        config.output_dir = 'output'
    if config.restart_training:
        shutil.rmtree(config.output_dir, ignore_errors=True)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    logger = setup_logger(os.path.join(config.output_dir, 'train_log'))
    logger.info(config.print())

    torch.manual_seed(config.seed)  # 为CPU设置随机种子
    if config.gpu_id is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")

    train_data = MyDataset(config.trainroot, data_shape=config.data_shape, n=config.n, m=config.m,
                           transform=transforms.ToTensor())
    train_loader = Data.DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=True,
                                   num_workers=int(config.workers), drop_last=True)
    writer = SummaryWriter(config.output_dir)
    model = PSENet(backbone=config.backbone, pretrained=config.pretrained, result_num=config.n, scale=config.scale)
    if not config.pretrained and not config.restart_training:
        model.apply(weights_init)
    ## loading the pretrained weights from drive
#     state_dict = torch.load(config.pretrained_path)
#     model.load_state_dict(state_dict)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    # dummy_input = torch.autograd.Variable(torch.Tensor(1, 3, 600, 800).to(device))
    # writer.add_graph(models=models, input_to_model=dummy_input)
    criterion = PSELoss(Lambda=config.Lambda, ratio=config.OHEM_ratio, reduction='mean')
    # optimizer = torch.optim.SGD(models.parameters(), lr=config.lr, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.checkpoint != '' and not config.restart_training:
        start_epoch = load_checkpoint(config.checkpoint, model, logger, device)
        start_epoch += 1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma,
                                                         last_epoch=start_epoch)
    else:
        start_epoch = config.start_epoch
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma)

    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_data.__len__(), all_step))
    epoch = 0
    try:
        for epoch in range(start_epoch, config.epochs):
            start = time.time()
            train_loss, lr = train_epoch(model, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,
                                         writer, logger)
            logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
                epoch, config.epochs, train_loss, time.time() - start, lr))
            # net_save_path = '{}/PSENet_{}_loss{:.6f}.pth'.format(config.output_dir, epoch,
            #                                                                               train_loss)
            # save_checkpoint(net_save_path, models, optimizer, epoch, logger)
            
            state_dict = model.state_dict()
            # replace the weight file
            filename = '{}/PSENet_resnet18.pth'.format(config.output_dir)
            if os.path.exists(filename):
                os.unlink(filename)
            torch.save(state_dict, filename)
        writer.close()
    except KeyboardInterrupt:
        save_checkpoint('{}/final.pth'.format(config.output_dir), model, optimizer, epoch, logger)


if __name__ == '__main__':
    main()
