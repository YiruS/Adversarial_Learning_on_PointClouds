#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import os
import sys
import pickle

import argparse
import numpy as np

from dataset.modelNetData import ModelNetDatasetGT, ModelNetDataset_noGT
from utils.utils import make_logger
from utils.trainer import run_training, run_training_semi, run_testing
from utils.model_utils import load_models
from utils.image_pool import ImagePool

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

RES_DIR = "/home/yirus/Projects/AdvSemiSeg/3D/results"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Transfer Learning for Eye Segmentation")
    parser.add_argument('name', type=str,
                        help="Name of the model for storing and loading purposes.")

    parser.add_argument("--pkl_file", type=str,
                        help="pickle file", )
    parser.add_argument("--train_file", type=str,
                    default="/home/yirus/Datasets/modelnet40_ply_hdf5_2048/train_files.txt",
                      help="data directory of Source dataset",)
    parser.add_argument("--test_file", type=str,
                    default="/home/yirus/Datasets/modelnet40_ply_hdf5_2048/test_files.txt",
                      help="data directory of Target dataset",)

    parser.add_argument("--batch_size",type=int,default=8, help="#data per batch")
    parser.add_argument("--lr",type=float, default=0.0001, help="lr for SS")
    parser.add_argument("--lr_D",type=float, default=0.0001, help="lr for D")
    parser.add_argument("--workers",type=int, default=0, help="#workers for dataloader")
    parser.add_argument("--loss",type=str, default="ce", help="Type of loss")
    parser.add_argument('--init_disc', type=str, default="xavier",
                        help='initialization of disc')

    parser.add_argument("--input_pts", type=int, default=1024, help="#pts per object")
    parser.add_argument("--disc_indim", type=int, default=40, help="#channeles to disc")
    parser.add_argument("--num_epochs", type=int, default=200, help="#epochs")
    parser.add_argument("--num_samples", type=int, default=10, help="#data w/ GT")
    parser.add_argument("--save_per_epoch", type=int, default=2, help="#epochs to save .pth")
    parser.add_argument("--test_epoch", type=int, default=2, help="#epochs to test")

    parser.add_argument('--pool_size', type=int, default=0,
                        help='buffer size for discriminator')
    parser.add_argument('--lambda_cls', type=float, default=1.0,
                        help='hyperparams for seg source')
    parser.add_argument('--lambda_adv', type=float, default=0.001,
                        help='hyperparams for adv of target')
    parser.add_argument('--lambda_semi', type=float, default=1.0,
                        help='hyperparams for semi target')
    parser.add_argument('--lambda_regu', type=float, default=0.001,
                        help='hyperparams for regulization')
    parser.add_argument('--semi_TH', type=float, default=0.8,
                        help='Threshold for semi')
    parser.add_argument('--semi_start_epoch', type=int, default=0,
                        help='Start epoch for semi')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pretrained model (.pth)')
    parser.add_argument('--train', action='store_true', default = False,
                        help = 'run training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='run testing')
    parser.add_argument('--run_semi', action='store_true', default=False,
                        help='run semi training')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='visulization with tensorboard')
    parser.add_argument('--adjust_lr', action='store_true', default=False,
                        help="adjust learning while training")
    parser.add_argument("--log_filename", type=str, default="default.log",
                        help="log file")

    args = parser.parse_args()
    return args


def main(args):
    print('===================================\n', )
    print("Root directory: {}".format(args.name))
    args.exp_dir = os.path.join(RES_DIR, args.name)
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    train_logger = make_logger("Train.log", args)
    test_logger = make_logger("Test.log", args)

    model = load_models(
        mode="cls",
        device=device,
        args=args,
    )
    model_D = load_models(
        mode="disc",
        device=device,
        args=args,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    optimizer.zero_grad()

    optimizer_D = optim.Adam(
        model_D.parameters(),
        lr=args.lr_D,
        betas=(0.9, 0.999),
    )
    optimizer_D.zero_grad()

    if args.tensorboard:
        writer = SummaryWriter(args.exp_dir)
    else:
        writer = None

    if (args.train or args.run_semi) and args.test:
        print("===================================")
        print("====== Loading Training Data ======")
        print("===================================")
        idx = np.arange(9840)
        np.random.shuffle(idx)
        sample_gt_list = idx[0:args.num_samples]
        sample_nogt_list = idx[args.num_samples:]
        filename = "gt_sample_{}.npy".format(args.name)
        np.save(filename, sample_gt_list)

        trainset_gt = ModelNetDatasetGT(
            root_list=args.train_file,
            sample_list=sample_gt_list,
        )
        trainset_nogt = ModelNetDataset_noGT(
            root_list=args.train_file,
            sample_list=sample_nogt_list,
        )

        trainloader_gt = torch.utils.data.DataLoader(
            trainset_gt,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        trainloader_nogt = torch.utils.data.DataLoader(
            trainset_nogt,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )

        print("===================================")
        print("====== Loading Test Data ======")
        print("===================================")
        testset = ModelNetDatasetGT(
            root_list=args.test_file,
            sample_list=None,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        args.iter_per_epoch = int(1.0 * trainset_gt.__len__() / args.batch_size)
        args.total_data = trainset_gt.__len__() + trainset_nogt.__len__()
        args.total_iterations = int(args.num_epochs *
                                    args.total_data / args.batch_size)
        # args.iter_save_epoch = args.save_per_epoch * args.iter_per_epoch
        # args.iter_test_epoch = args.test_epoch * args.iter_per_epoch
        # args.semi_start = int(args.semi_start_epoch *
        #                                trainset_gt.__len__() / args.batch_size)
        args.iter_save_epoch = args.save_per_epoch * int(args.total_data / args.batch_size)
        args.iter_test_epoch = args.test_epoch * int(args.total_data / args.batch_size)
        args.semi_start = int(args.semi_start_epoch *
                              args.total_data / args.batch_size)

    if (args.train or args.run_semi) and args.test:
        model.train()
        model_D.train()

        model.to(args.device)
        model_D.to(args.device)

        #class_weight = 1.0 / trainset_gt.get_class_probability().to(device)

        cls_loss = torch.nn.CrossEntropyLoss().to(device)
        gan_loss = torch.nn.BCEWithLogitsLoss().to(device)
        semi_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

        history_pool_gt = ImagePool(args.pool_size)
        history_pool_nogt = ImagePool(args.pool_size)

        trainloader_gt_iter = enumerate(trainloader_gt)
        targetloader_nogt_iter = enumerate(trainloader_nogt)


        if args.train:
            run_training(
                trainloader_gt=trainloader_gt,
                trainloader_nogt=trainloader_nogt,
                trainloader_gt_iter=trainloader_gt_iter,
                targetloader_nogt_iter=targetloader_nogt_iter,
                testloader=testloader,
                model=model,
                model_D=model_D,
                gan_loss=gan_loss,
                cls_loss=cls_loss,
                optimizer=optimizer,
                optimizer_D=optimizer_D,
                history_pool_gt=history_pool_gt,
                history_pool_nogt=history_pool_nogt,
                writer=writer,
                train_logger=train_logger,
                test_logger=test_logger,
                args=args,
            )
        elif args.run_semi:
            run_training_semi(
                trainloader_gt=trainloader_gt,
                trainloader_nogt=trainloader_nogt,
                trainloader_gt_iter=trainloader_gt_iter,
                targetloader_nogt_iter=targetloader_nogt_iter,
                testloader=testloader,
                model=model,
                model_D=model_D,
                gan_loss=gan_loss,
                cls_loss=cls_loss,
                semi_loss=semi_loss,
                optimizer=optimizer,
                optimizer_D=optimizer_D,
                history_pool_gt=history_pool_gt,
                history_pool_nogt=history_pool_nogt,
                writer=writer,
                train_logger=train_logger,
                test_logger=test_logger,
                args=args,
            )

    if args.test:
        print("===================================")
        print("====== Loading Testing Data =======")
        print("===================================")
        testset = ModelNetDatasetGT(
            root_list=args.test_file,
            sample_list=None,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        criterion = torch.nn.CrossEntropyLoss().to(device)

        run_testing(
            dataloader=testloader,
            model=model,
            criterion=criterion,
            logger=test_logger,
            test_iter=100000000,
            writer=None,
            args=args,
        )

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
