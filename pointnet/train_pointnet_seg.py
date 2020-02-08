#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import os
import sys

import argparse
import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)

from dataset.shapeNetData import ShapeNetDatasetGT, ShapeNetDataset_noGT
from utils.utils import make_logger
from utils.trainer import run_training_pointnet_seg, run_testing_seg
from utils.model_utils import load_models


import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

RES_DIR = "/home/yirus/Projects/Adversarial_Learning_on_PointClouds/pointnet/seg"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Transfer Learning for Eye Segmentation")
    parser.add_argument('name', type=str,
                        help="Name of the model for storing and loading purposes.")

    parser.add_argument("--gt_sample_list", type=str,
                        help="GT sample file", )
    parser.add_argument("--train_file", type=str,
                    default="/home/yirus/Datasets/shapeNet/hdf5_data/train_hdf5_file_list.txt",
                      help="data directory of Source dataset",)
    parser.add_argument("--test_file", type=str,
                    default="/home/yirus/Datasets/shapeNet/hdf5_data/test_hdf5_file_list.txt",
                      help="data directory of Target dataset",)

    parser.add_argument("--batch_size", type=int,default=16, help="#data per batch")
    parser.add_argument("--lr", type=float, default=0.0001, help="lr for SS")
    parser.add_argument("--workers", type=int, default=0, help="#workers for dataloader")
    parser.add_argument("--loss", type=str, default="ce", help="Type of loss")

    parser.add_argument("--input_pts", type=int, default=2048, help="#pts per object")
    parser.add_argument("--num_epochs", type=int, default=200, help="#epochs")
    parser.add_argument("--save_per_epoch", type=int, default=5, help="#epochs to save .pth")
    parser.add_argument("--test_epoch", type=int, default=5, help="#epochs to test")

    parser.add_argument('--lambda_seg', type=float, default=1.0,
                        help='hyperparams for seg source')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pretrained model (.pth)')
    parser.add_argument('--train', action='store_true', default = False,
                        help = 'run training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='run testing')
    parser.add_argument('--tsne', action='store_true', default=False,
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
        mode="seg",
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

    if args.train and args.test:
        print("===================================")
        print("====== Loading Training Data ======")
        print("===================================")

        if args.gt_sample_list != None:
            sample_gt_list = np.load(args.gt_sample_list)
        else:
            sample_gt_list = None

        trainset_gt = ShapeNetDatasetGT(
            root_list=args.train_file,
            sample_list=sample_gt_list,
            num_classes=16,
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
        testset = ShapeNetDatasetGT(
            root_list=args.test_file,
            sample_list=None,
            num_classes=16,
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

        seg_loss = torch.nn.CrossEntropyLoss().to(device)

        trainloader_gt_iter = enumerate(trainloader_gt)

        run_training_pointnet_seg(
            trainloader_gt=trainloader_gt,
            trainloader_gt_iter=trainloader_gt_iter,
            testloader=testloader,
            testdataset=testset,
            model=model,
            seg_loss=seg_loss,
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
        testset = ShapeNetDatasetGT(
            root_list=args.test_file,
            sample_list=None,
            num_classes=16,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        criterion = torch.nn.CrossEntropyLoss().to(device)

        run_testing_seg(
            dataloader=testloader,
            dataset=testset,
            model=model,
            criterion=criterion,
            logger=test_logger,
            test_iter=100000000,
            writer=None,
            args=args,
        )

    if args.tsne:
        from utils.metric import batch_get_iou, object_names
        from torch.autograd import Variable

        args.batch_size = 1

        labels = []
        objects = []

        if args.gt_sample_list != None:
            sample_gt_list = np.load(args.gt_sample_list)
        else:
            sample_gt_list = None

        trainset_gt = ShapeNetDatasetGT(
            root_list=args.train_file,
            sample_list=sample_gt_list,
            num_classes=16,
        )

        trainset_nogt = ShapeNetDataset_noGT(
            root_list=args.train_file,
            sample_list=sample_gt_list,
            num_classes=16,
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

        # print("===================================")
        # print("====== Loading Test Data ======")
        # print("===================================")
        # testset = ShapeNetDatasetGT(
        #     root_list=args.test_file,
        #     sample_list=None,
        #     num_classes=16,
        # )
        # testloader = torch.utils.data.DataLoader(
        #     testset,
        #     batch_size=args.batch_size,
        #     shuffle=False,
        #     num_workers=args.workers,
        #     pin_memory=True,
        # )
        #
        # print("===================================")
        # print("====== Loading Testing Data =======")
        # print("===================================")
        # testset = ShapeNetDatasetGT(
        #     root_list=args.test_file,
        #     sample_list=None,
        #     num_classes=16,
        # )
        # testloader = torch.utils.data.DataLoader(
        #     testset,
        #     batch_size=args.batch_size,
        #     shuffle=False,
        #     num_workers=args.workers,
        #     pin_memory=True,
        # )

        model.eval()

        shape_ious = np.empty(len(object_names), dtype=np.object)
        for i in range(shape_ious.shape[0]):
            shape_ious[i] = []


        all_shapes_train_gt = np.empty((len(trainset_gt), 2048))
        for batch_idx, data in enumerate(trainloader_gt):
            pts, cls, seg = data
            pts, cls, seg = Variable(pts).float(), \
                            Variable(cls), Variable(seg).type(torch.LongTensor)
            pts, cls, seg = pts.to(args.device), cls.to(args.device), seg.long().to(args.device)

            labels.append(1)
            objects.append(int(cls.argmax(axis=2).squeeze().cpu().numpy()))

            with torch.set_grad_enabled(False):
                pred, global_shape = model(pts, cls)

            all_shapes_train_gt[batch_idx,:] = global_shape.squeeze().detach().cpu().numpy()

        all_shapes_train_nogt = np.empty((len(trainset_nogt), 2048))
        for batch_idx, data in enumerate(trainloader_nogt):
            pts, cls = data
            pts, cls = Variable(pts).float(), Variable(cls)
            pts, cls = pts.to(args.device), cls.to(args.device)

            with torch.set_grad_enabled(False):
                pred, global_shape = model(pts, cls)

            all_shapes_train_nogt[batch_idx, :] = global_shape.squeeze().detach().cpu().numpy()

            labels.append(0)
            objects.append(int(cls.argmax(axis=2).squeeze().cpu().numpy()))

        all_shapes = np.concatenate((all_shapes_train_gt, all_shapes_train_nogt), axis=0)

        np.save("global_shape_{}.npy".format(len(trainset_gt)), all_shapes)

        # from MulticoreTSNE import MulticoreTSNE as TSNE
        # import matplotlib.pyplot as plt
        #
        # tsne = TSNE(n_jobs=8)
        # embeddings = tsne.fit_transform(all_shapes)
        # vis_x = embeddings[:, 0]
        # vis_y = embeddings[:, 1]
        # plt.scatter(vis_x, vis_y, c=objects, cmap=plt.cm.get_cmap("jet", 10), marker='.')
        # plt.colorbar(ticks=range(10))
        # plt.clim(-0.5, 9.5)
        # plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
