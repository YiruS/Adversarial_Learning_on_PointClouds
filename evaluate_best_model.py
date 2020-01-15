#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import os
import sys
import pickle

import argparse
import glob
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

    test_logger = make_logger("Test.log", args)

    model = load_models(
        mode="cls",
        device=device,
        args=args,
    )
    # model_D = load_models(
    #     mode="disc",
    #     device=device,
    #     args=args,
    # )

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    optimizer.zero_grad()

    #optimizer_D = optim.Adam(
    #     model_D.parameters(),
    #     lr=args.lr_D,
    #     betas=(0.9, 0.999),
    # )
    # optimizer_D.zero_grad()

    if args.tensorboard:
        writer = SummaryWriter(args.exp_dir)
    else:
        writer = None


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

    train_pth = glob.glob(args.exp_dir+"/model_train_epoch_*.pth")
    print("Total #models: {}".format(len(train_pth)))

    max_accu = float("-inf")
    for i in range(len(train_pth)):
        print("Processing {} .....".format(i))
        print("model {} .........".format(train_pth[i]))
        model.load_state_dict(torch.load(train_pth[i]))

        curr_accu, curr_loss = run_testing(
            dataloader=testloader,
            model=model,
            criterion=criterion,
            logger=test_logger,
            test_iter=100000000,
            writer=None,
            args=args,
        )

        if curr_accu > max_accu:
            max_accu = curr_accu
            max_pth = train_pth[i]

    print("Max accuracy: {:.4f}".format(max_accu))
    print("Trained model: {}".format(max_pth))
if __name__ == "__main__":
    args = parse_arguments()
    main(args)