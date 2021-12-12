import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import random
import math

import cv2
import os
import warnings
import numpy as np

from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet152, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder

import torch.nn.functional as F
from test import TTAFrame


if __name__ == '__main__':

    # the network need the size to be a multiple of 32, resize is intriduced
    ORIG_SHAPE = (400, 400)
    SHAPE = (512, 512)
    NAME = 'dinknet152'
    BATCHSIZE_PER_CARD = 4

    train_root = 'dataset/train/'
    image_root = os.path.join(train_root, 'images')
    gt_root = os.path.join(train_root, 'groundtruth')
    image_list = np.array(sorted(
        [f for f in os.listdir(image_root) if f.endswith('.png')]))
    gt_list = np.array(sorted(
        [f for f in os.listdir(gt_root) if f.endswith('.png')]))
    # imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(train_root))

    # random select 20% of training data for validation
    total_data_num = image_list.shape[0]
    validation_data_num = math.ceil(total_data_num * 0.2)
    validation_idx = random.sample(range(total_data_num), validation_data_num)
    new_train_indx = list(
        set(range(total_data_num)).difference(set(validation_idx)))

    val_img_list = image_list[validation_idx].tolist()
    val_gt_list = gt_list[validation_idx].tolist()
    image_list = image_list[new_train_indx].tolist()
    gt_list = gt_list[new_train_indx].tolist()
    # trainlist = image_list

    # val_root = 'D:/complete_project/Dinknet/road512/val/'
    # imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(val_root))
    # vallist = map(lambda x: x[:-8], imagelist)
    # vallist = list(vallist)

    # solver = MyFrame(DinkNet34, dice_bce_loss, 1e-3)
    solver = MyFrame(DinkNet152, dice_bce_loss, 1e-3)
    # solver.load('./weights/test.th')

    if torch.cuda.is_available():
        train_batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        val_batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
    else:
        train_batchsize = BATCHSIZE_PER_CARD
        val_batchsize = BATCHSIZE_PER_CARD

    #  data preprocessing here
    train_dataset = ImageFolder(image_list, image_root, gt_root, SHAPE)
    val_dataset = ImageFolder(val_img_list, image_root, gt_root, SHAPE)
    # print("train_dataset", train_dataset)

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batchsize,
        shuffle=True,
        num_workers=0)
    # print("data loader", data_loader)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batchsize,
        shuffle=True,
        num_workers=0)

    mylog = open('logs/'+NAME+'.log', 'w')
    tic = time()
    # device = torch.device('cuda:0')
    no_optim = 0
    total_epoch = 300
    train_epoch_best_loss = 100.

    for epoch in range(1, total_epoch + 1):
        print('---------- Epoch:'+str(epoch) + ' ----------')
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0

        print('Train:')
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)

        duration_of_epoch = int(time()-tic)

        if epoch % 5 == 0 and os.path.exists('weights/'+NAME+'.th'):
            val_data_loader_iter = iter(val_data_loader)
            validation_epoch_loss = 0
            print("Validation: ")
            for val_img, val_mask in val_data_loader_iter:
                solver.set_input(val_img, val_mask)
                val_loss = solver.optimize(True)
                validation_epoch_loss += val_loss
            validation_epoch_loss /= len(val_img_list)
            print('--epoch:', epoch,  '  --validation_loss:',
                  validation_epoch_loss)

        mylog.write('********************' + '\n')
        mylog.write('--epoch:' + str(epoch) + '  --time:' + str(duration_of_epoch) + '  --train_loss:' + str(
            train_epoch_loss) + '\n')
        print('--epoch:', epoch, '  --time:', duration_of_epoch, '  --train_loss:',
              train_epoch_loss)
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            # last_save_name = 'weights/' + NAME + \
            #     '_{:.5f}.th'.format(train_epoch_loss)
            # solver.save(last_save_name)
            solver.save('weights/'+NAME+'.th')

        if no_optim > 6:
            print(mylog, 'early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            # solver.load(last_save_name)
            solver.load('weights/'+NAME+'.th')
            solver.update_lr(5.0, factor=True, mylog=mylog)
        mylog.flush()

    print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()
