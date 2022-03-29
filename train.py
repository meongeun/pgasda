import sys
import time
import cv2

from data import create_dataloader
from utils import dataset_util, util
from models import create_model
from matplotlib import pyplot as plt
from utils.util import SaveResults
from options.train_options import TrainOptions

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from data.datasets import get_dataset, ConcatDataset
from data.transform import RandomImgAugment, DepthToTensor, PoseToTensor


import numpy as np
import torch.nn as nn

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


if __name__ == '__main__':
    opt = TrainOptions().parse()

    src_train_loader, tgt_train_loader = create_dataloader(opt)
    print('#training images = %d' %(len(src_train_loader)+len(tgt_train_loader)))

    model = create_model(opt)
    model.setup(opt)
    save_results = SaveResults(opt)

    total_steps = 0
    # for i in tgt_train_loader:
    #     print(i.keys())
    #     exit()

    lr = opt.lr_task
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # training
        print("training stage (epoch: %s) starting...................." % epoch)
        src_train_loader_iterator = iter(src_train_loader)
        tgt_train_loader_iterator = iter(tgt_train_loader)
        ind = -1
        while True:
            ind+=1
            try:
                src_data = next(src_train_loader_iterator)
                tgt_data = next(tgt_train_loader_iterator)
                # print('hi2', tgt_data)
            except StopIteration:
                break

            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(src_data['src'], tgt_data['tgt'])
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                save_results.print_current_losses(epoch, epoch_iter, lr, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, total_steps))
                model.save_networks('latest')

            if total_steps % opt.save_result_freq == 0:
                save_results.save_current_results(model.get_current_visuals(), epoch)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))

            model.save_networks('latest')
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        lr = model.update_learning_rate()

