"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

Machine Learning paper
Distribution Aligned Semantics Adaption for Lifelong Person Re-Identification
URL: https://arxiv.org/abs/2405.19695
GitHub: https://github.com/QizaoWang/DASA-LReID
"""

from __future__ import print_function, absolute_import
import argparse
import datetime
import os.path as osp
import sys
import time
import numpy as np

import torch
from torch.backends import cudnn
import torch.nn as nn
import random

from datasets import get_data
from utils.logging import Logger
from utils.serialization import save_checkpoint
from utils.metrics import R1_mAP_eval
from utils.lr_scheduler import WarmupMultiStepLR
from utils.my_tools import eval_func
from models.resnet import SA, ResNet
from models.layers import DataParallel
from trainer import Trainer


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    if args.training_order == 1:
        names = ['market1501', 'dukemtmc', 'cuhk_sysu', 'msmt17']
    elif args.training_order == 2:
        names = ['viper', 'market1501', 'cuhk_sysu', 'msmt17']

    log_name = 'log.txt'
    sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset_market, num_classes_market, train_loader_market, test_loader_market, init_loader_market, sampler_market = \
        get_data(names[0], args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_duke, num_classes_duke, train_loader_duke, test_loader_duke, init_loader_duke, sampler_duke = \
        get_data(names[1], args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_cuhksysu, num_classes_cuhksysu, train_loader_cuhksysu, test_loader_cuhksysu, init_loader_cuhksysu, sampler_cuhksysu = \
        get_data(names[2], args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_msmt17, num_classes_msmt17, train_loader_msmt17, test_loader_msmt17, init_loader_msmt17, sampler_msmt17 = \
        get_data(names[3], args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    datasets = [dataset_market, dataset_duke, dataset_cuhksysu, dataset_msmt17]
    num_class_list = [num_classes_market, num_classes_duke, num_classes_cuhksysu, num_classes_msmt17]
    train_loaders = [train_loader_market, train_loader_duke, train_loader_cuhksysu, train_loader_msmt17]
    test_loaders = [test_loader_market, test_loader_duke, test_loader_cuhksysu, test_loader_msmt17]
    evaluators = [R1_mAP_eval(len(dataset_market.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_duke.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_cuhksysu.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_msmt17.query), max_rank=50, feat_norm=True)]

    # Create model
    model = ResNet(last_stride=1, num_class=num_class_list)
    state_dict = torch.load(args.pt_weight)
    model.load_state_dict(state_dict, strict=False)

    # freeze pre-trained model parameters except BN, SA, and classifier
    for n, p in model.named_parameters():
        if 'bn' in n or 'downsample.1' in n:
            p.requires_grad = True
        elif 'classifier' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    model.bn.bias.requires_grad = False

    def add_SA(model):
        for name, layer in model.named_children():
            if isinstance(layer, nn.Conv2d):
                model._modules[name] = SA(layer, kernel_size=5)
            else:
                add_SA(layer)
        return model

    model = add_SA(model)

    model = DataParallel(model.cuda())
    trainer = Trainer(model)

    print('Continual training starts!')
    end = time.time()

    training_phase = 0
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, [30], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    for epoch in range(0, args.epochs):
        trainer.train(epoch + 1, train_loaders[training_phase], optimizer, lr_scheduler, training_phase,
                      train_iters=150 if names[0] == 'viper' else len(train_loaders[training_phase]))

        if (epoch + 1) % 10 == 0:
            cmc, mAP = eval_func(epoch + 1, evaluators[training_phase], model,
                                 test_loaders[training_phase], names[training_phase])

            if args.save_checkpoint:
                save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'mAP': mAP,
                    'rank1': cmc[0],
                }, True, fpath=osp.join(args.logs_dir, names[training_phase] + '_ep' + str(epoch + 1) + '.pth'))
    print()  # add a blank line

    for training_phase in range(1, len(datasets)):
        params = []
        for key, value in model.named_params(model):
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(params)
        lr_scheduler = WarmupMultiStepLR(optimizer, [10], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

        for epoch in range(0, args.epochs):
            trainer.train(epoch + 1, train_loaders[training_phase], optimizer, lr_scheduler, training_phase,
                          train_iters=len(train_loaders[training_phase]))
            if (epoch + 1) % 10 == 0:
                cmc, mAP = eval_func(epoch + 1, evaluators[training_phase], model,
                                     test_loaders[training_phase], names[training_phase])

                if args.save_checkpoint:
                    save_checkpoint({
                        'state_dict': model.module.state_dict(),
                        'mAP': mAP,
                        'rank1': cmc[0],
                    }, True, fpath=osp.join(args.logs_dir, names[training_phase] + '_ep' + str(epoch + 1) + '.pth'))
        print()  # add a blank line

    total_time = round(time.time() - end)
    total_time = str(datetime.timedelta(seconds=total_time))
    print('finished, time:{}'.format(total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=2,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    # training configs
    parser.add_argument('--training-order', type=int, default=1)
    parser.add_argument('--pt-weight', type=str, default='lupws_r50.pth', metavar='PATH',
                        help="path of the pre-trained weight")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--seed', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--save_checkpoint', action='store_true', help='save model checkpoint')

    main()
