#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import os
import sys

import argparse
import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)

from dataset.modelNetData import ModelNetDatasetGT
from utils.utils import make_logger
from utils.trainer import run_training_pointnet_cls, run_testing
from utils.model_utils import load_models


import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

RES_DIR = "/home/yirus/Projects/Adversarial_Learning_on_PointClouds/pointnet"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Transfer Learning for Eye Segmentation")
    parser.add_argument('name', type=str,
                        help="Name of the model for storing and loading purposes.")

    parser.add_argument("--gt_sample_list", type=str,
                        help="GT sample file", )
    parser.add_argument("--train_file", type=str,
                    default="/home/yirus/Datasets/modelnet40_ply_hdf5_2048/train_files.txt",
                      help="data directory of Source dataset",)
    parser.add_argument("--test_file", type=str,
                    default="/home/yirus/Datasets/modelnet40_ply_hdf5_2048/test_files.txt",
                      help="data directory of Target dataset",)

    parser.add_argument("--batch_size", type=int,default=16, help="#data per batch")
    parser.add_argument("--lr", type=float, default=0.0001, help="lr for SS")
    parser.add_argument("--workers", type=int, default=0, help="#workers for dataloader")
    parser.add_argument("--loss", type=str, default="ce", help="Type of loss")

    parser.add_argument("--input_pts", type=int, default=1024, help="#pts per object")
    parser.add_argument("--num_epochs", type=int, default=200, help="#epochs")
    parser.add_argument("--save_per_epoch", type=int, default=2, help="#epochs to save .pth")
    parser.add_argument("--test_epoch", type=int, default=2, help="#epochs to test")

    parser.add_argument('--lambda_cls', type=float, default=1.0,
                        help='hyperparams for seg source')
    parser.add_argument('--lambda_regu', type=float, default=0.001,
                        help='hyperparams for regulization')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pretrained model (.pth)')
    parser.add_argument('--train', action='store_true', default = False,
                        help = 'run training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='run testing')
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

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    optimizer.zero_grad()

    if args.tensorboard:
        writer = SummaryWriter(args.exp_dir)
    else:
        writer = None

    if (args.train or args.run_semi) and args.test:
        print("===================================")
        print("====== Loading Training Data ======")
        print("===================================")

        sample_gt_list = np.load(args.gt_sample_list)

        trainset_gt = ModelNetDatasetGT(
            root_list=args.train_file,
            sample_list=sample_gt_list,
        )

        trainloader_gt = torch.utils.data.DataLoader(
            trainset_gt,
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

        args.iter_per_epoch = int(trainset_gt.__len__() / args.batch_size)
        args.total_data = trainset_gt.__len__()
        args.total_iterations = int(args.num_epochs * args.total_data / args.batch_size)
        args.iter_save_epoch = args.save_per_epoch * int(args.total_data / args.batch_size)
        args.iter_test_epoch = args.test_epoch * int(args.total_data / args.batch_size)

    if args.train and args.test:
        model.train()
        model.to(args.device)

        cls_loss = torch.nn.CrossEntropyLoss().to(device)

        trainloader_gt_iter = enumerate(trainloader_gt)

        run_training_pointnet_cls(
            trainloader_gt=trainloader_gt,
            trainloader_gt_iter=trainloader_gt_iter,
            testloader=testloader,
            model=model,
            cls_loss=cls_loss,
            optimizer=optimizer,
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
