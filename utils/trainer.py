#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import os
import sys

import numpy as np
import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)
sys.path.append("%s/../.." % file_path)

from utils.utils import make_D_label, adjust_learning_rate
from models.pointnet import feature_transform_regularizer

INPUT_CHANNELS = 3
NUM_CLASSES = 4


def run_testing(
        dataloader,
        model,
        criterion,
        logger,
        test_iter,
        writer,
        args,
):
    model.eval()

    total_accuracy = 0.0
    total_loss = 0.0
    for batch_idx, data in tqdm.tqdm(enumerate(dataloader), total=dataloader.__len__()):
        pts, cls = data
        pts, cls = Variable(pts).float(), \
                      Variable(cls).type(torch.LongTensor)
        pts, cls = pts.to(args.device), cls.to(args.device)

        with torch.set_grad_enabled(False):
            pred, _, _ = model(pts)
            loss = criterion(pred, cls)

        cls = cls.detach().cpu().numpy()
        pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
        accu = sum(np.equal(pred, cls))
        total_accuracy += accu
        total_loss += loss.item()

    logger.info("Test accuracy: {:.4f} loss: {:.3f}".format(
        total_accuracy / float(args.batch_size*dataloader.__len__()),
        total_loss / float(args.batch_size*dataloader.__len__()),
    ))

    if args.tensorboard and (writer is not None):
        writer.add_scalar('Loss/test_cls', total_loss / float(args.batch_size*dataloader.__len__()), test_iter)
        writer.add_scalar('Accuracy/test', total_accuracy / float(args.batch_size*dataloader.__len__()), test_iter)

    return total_accuracy / float(args.batch_size*dataloader.__len__()), \
           total_loss / float(args.batch_size*dataloader.__len__())

def run_testing_seg(
        dataloader,
        dataset,
        model,
        criterion,
        logger,
        test_iter,
        writer,
        args,
):
    model.eval()

    total_accuracy = 0.0
    total_loss = 0.0
    for batch_idx, data in tqdm.tqdm(enumerate(dataloader), total=dataloader.__len__()):
        pts, cls, seg = data
        pts, cls, seg = Variable(pts).float(), \
                      Variable(cls), Variable(seg).type(torch.LongTensor)
        pts, cls, seg = pts.to(args.device), cls.to(args.device), seg.long().to(args.device)

        with torch.set_grad_enabled(False):
            pred, _ = model(pts, cls)
            loss = criterion(pred, seg)

        pred_seg = pred.max(1)[1]
        accu = pred_seg.eq(seg).cpu().numpy().sum()
        total_accuracy += accu / float(args.input_pts)
        total_loss += loss.item()

    logger.info("Test accuracy: {:.4f} loss: {:.3f}".format(
        total_accuracy / float(len(dataset)),
        total_loss / float(len(dataset)),
    ))

    if args.tensorboard and (writer is not None):
        writer.add_scalar('Loss/test_cls',
                          total_loss / float(len(dataset)), test_iter)
        writer.add_scalar('Accuracy/test',
                          total_accuracy / float(len(dataset)), test_iter)

    return total_accuracy / float(len(dataset)), \
           total_loss / float(len(dataset))


def run_training_pointnet_cls(
    trainloader_gt,
    trainloader_gt_iter,
    testloader,
    model,
    cls_loss,
    optimizer,
    train_logger,
    test_logger,
    writer,
    args,
):
    max_test_accu = float("-inf")

    for i_iter in range(args.total_iterations):
        loss_cls_value = 0
        loss_regulization = 0

        model.train()
        optimizer.zero_grad()


        ## train with points w/ GT ##
        try:
            _, batch = next(trainloader_gt_iter)
        except StopIteration:
            trainloader_gt_iter = enumerate(trainloader_gt)
            _, batch = next(trainloader_gt_iter)

        pts, cls = batch
        pts, cls = pts.to(args.device), cls.long().to(args.device)

        pred, global_gt, high_feat = model(pts)
        l = cls_loss(pred, cls)
        loss_cls_value += l.item()
        if high_feat is not None:
            l_regu = feature_transform_regularizer(high_feat)
            loss_regulization += l_regu.item()
        else:
            l_regu = None

        if l_regu is None:
            loss = args.lambda_cls * l
        else:
            loss = args.lambda_cls * l + \
                   args.lambda_regu * l_regu
        loss.backward()
        optimizer.step()

        train_logger.info('iter = {0:8d}/{1:8d} '
              'loss_cls = {2:.3f} '
              'loss regu = {3:.3f} '.format(
                i_iter, args.total_iterations,
                loss_cls_value,
                loss_regulization,
            )
        )

        if args.tensorboard:
            writer.add_scalar('Loss/train_cls', loss_cls_value, i_iter)

        if i_iter % args.iter_save_epoch == 0:
            curr_epoch = i_iter // trainloader_gt.__len__()
            torch.save(model.state_dict(), os.path.join(args.exp_dir,
                                                       "model_train_epoch_{}.pth").format(curr_epoch))

        if i_iter % args.iter_test_epoch == 0:
            curr_accu, curr_loss = run_testing(
                dataloader=testloader,
                model=model,
                criterion=cls_loss,
                logger=test_logger,
                test_iter=i_iter,
                writer=writer,
                args=args,
            )
            if max_test_accu < curr_accu:
                max_test_accu = curr_accu
                max_train_epoch = i_iter // args.iter_test_epoch
                torch.save(model.state_dict(), os.path.join(args.exp_dir,
                                                            "model_train_best.pth"))

    if args.tensorboard:
        writer.close()

    train_logger.info("Max test accuracy: {:.4f}".format(max_test_accu))
    train_logger.info("Train model is at epoch: {}".format(max_train_epoch))

def run_training_pointnet_seg(
    trainloader_gt,
    trainloader_gt_iter,
    testloader,
    testdataset,
    model,
    seg_loss,
    optimizer,
    train_logger,
    test_logger,
    writer,
    args,
):
    max_test_accu = float("-inf")

    for i_iter in range(args.total_iterations):
        loss_seg_value = 0

        model.train()
        optimizer.zero_grad()

        ## train with points w/ GT ##
        try:
            _, batch = next(trainloader_gt_iter)
        except StopIteration:
            trainloader_gt_iter = enumerate(trainloader_gt)
            _, batch = next(trainloader_gt_iter)

        pts, cls, seg = batch
        pts, cls, seg = pts.to(args.device), cls.to(args.device), seg.long().to(args.device)

        pred, global_gt = model(pts, cls)
        l = seg_loss(pred, seg)
        loss_seg_value += l.item()

        loss = args.lambda_seg * l
        loss.backward()
        optimizer.step()

        train_logger.info('iter = {0:8d}/{1:8d} '
              'loss_seg = {2:.3f} '.format(
                i_iter, args.total_iterations,
                loss_seg_value,
            )
        )

        if args.tensorboard:
            writer.add_scalar('Loss/train_seg', loss_seg_value, i_iter)

        if i_iter % args.iter_save_epoch == 0:
            curr_epoch = i_iter // trainloader_gt.__len__()
            torch.save(model.state_dict(), os.path.join(args.exp_dir,
                                                       "model_train_epoch_{}.pth").format(curr_epoch))

        if i_iter % args.iter_test_epoch == 0:
            curr_accu, curr_loss = run_testing_seg(
                dataset=testdataset,
                dataloader=testloader,
                model=model,
                criterion=seg_loss,
                logger=test_logger,
                test_iter=i_iter,
                writer=writer,
                args=args,
            )
            if max_test_accu < curr_accu:
                max_test_accu = curr_accu
                max_train_epoch = i_iter // args.iter_test_epoch
                torch.save(model.state_dict(), os.path.join(args.exp_dir,
                                                            "model_train_best.pth"))

    if args.tensorboard:
        writer.close()

    train_logger.info("Max test accuracy: {:.4f}".format(max_test_accu))
    train_logger.info("Train model is at epoch: {}".format(max_train_epoch))


def run_training(
    trainloader_gt,
    trainloader_nogt,
    trainloader_gt_iter,
    targetloader_nogt_iter,
    testloader,
    model,
    model_D,
    gan_loss,
    cls_loss,
    optimizer,
    optimizer_D,
    history_pool_gt,
    history_pool_nogt,
    train_logger,
    test_logger,
    writer,
    args,
):
    gt_label = 1
    nogt_label = 0
    max_test_accu = float("-inf")

    for i_iter in range(args.total_iterations):
        loss_cls_value = 0
        loss_adv_value = 0
        loss_regulization = 0
        loss_D_value = 0

        model.train()
        model_D.train()

        optimizer.zero_grad()
        optimizer_D.zero_grad()

        # adjust_learning_rate(
        #     optimizer=optimizer_SS,
        #     learning_rate=args.lr_SS,
        #     i_iter=i_iter,
        #     max_steps=args.total_iterations,
        #     power=0.9,
        # )
        #
        # adjust_learning_rate(
        #     optimizer=optimizer_D,
        #     learning_rate=args.lr_D,
        #     i_iter=i_iter,
        #     max_steps=args.total_iterations,
        #     power=0.9,
        # )

        ## train G ##
        for param in model_D.parameters():
            param.requires_grad = False

        ## train with points w/ GT ##
        try:
            _, batch = next(trainloader_gt_iter)
        except StopIteration:
            trainloader_gt_iter = enumerate(trainloader_gt)
            _, batch = next(trainloader_gt_iter)

        pts, cls = batch
        pts, cls = pts.to(args.device), cls.long().to(args.device)

        pred, global_gt, high_feat = model(pts)
        l = cls_loss(pred, cls)
        loss_cls_value += l.item()
        # global_softmax = F.log_softmax(global_gt, dim=1)
        pred_gt_softmax = F.log_softmax(pred, dim=1)
        # if high_feat is not None:
        #     l_regu = feature_transform_regularizer(high_feat)
        #     loss_regulization += l_regu.item()
        # else:
        #     l_regu = None


        ## train with target ##
        try:
            _, batch = next(targetloader_nogt_iter)
        except StopIteration:
            targetloader_nogt_iter = enumerate(trainloader_nogt)
            _, batch = next(targetloader_nogt_iter)

        pts_nogt = batch
        pts_nogt = pts_nogt.to(args.device)

        pred_nogt, global_nogt, high_feat = model(pts_nogt)
        # global_nogt_softmax = F.log_softmax(global_nogt, dim=1)
        pred_nogt_softmax = F.log_softmax(pred_nogt, dim=1)
        # if high_feat is not None:
        #     l_regu = feature_transform_regularizer(high_feat)
        #     loss_regulization += l_regu.item()
        # else:
        #     l_regu = None

        D_out = model_D(pred_nogt_softmax)  #global_nogt_softmax
        generated_label = make_D_label(
            input=D_out,
            value=gt_label,
            device=args.device,
            random=False,
        )

        loss_adv = gan_loss(D_out, generated_label)
        loss_adv_value += loss_adv.item()

        loss = args.lambda_cls * l + \
               args.lambda_adv * loss_adv

        # if l_regu is None:
        #     loss = args.lambda_cls * l + \
        #            args.lambda_adv * loss_adv
        # else:
        #     loss = args.lambda_cls * l + \
        #            args.lambda_adv * loss_adv + \
        #            args.lambda_regu * l_regu
        loss.backward()

        ## train D ##
        for param in model_D.parameters():
            param.requires_grad = True

        ## train w/ GT ##
        # global_softmax = global_softmax.detach()
        pred_gt_softmax = pred_gt_softmax.detach()
        pool_gt = history_pool_gt.query(pred_gt_softmax)
        D_out = model_D(pool_gt)
        generated_label = make_D_label(
            input=D_out,
            value=gt_label,
            device=args.device,
            random=True,
        )
        loss_D = gan_loss(D_out, generated_label)
        loss_D = loss_D * 0.5
        loss_D.backward()
        loss_D_value += loss_D.item()

        ## train wo GT ##
        # global_nogt_softmax = global_nogt_softmax.detach()
        pred_nogt_softmax = pred_nogt_softmax.detach()
        pool_nogt = history_pool_nogt.query(pred_nogt_softmax)
        D_out = model_D(pool_nogt)
        generated_label = make_D_label(
            input=D_out,
            value=nogt_label,
            device=args.device,
            random=True,
        )
        loss_D = gan_loss(D_out, generated_label)
        loss_D = loss_D * 0.5
        loss_D.backward()
        loss_D_value += loss_D.item()

        optimizer.step()
        optimizer_D.step()

        train_logger.info('iter = {0:8d}/{1:8d} '
              'loss_cls = {2:.3f} '
              'loss_adv = {3:.3f} '
              # 'loss regu = {4:.3f} '
              'loss_D = {4:.3f}'.format(
                i_iter, args.total_iterations,
                loss_cls_value,
                loss_adv_value,
                # loss_regulization,
                loss_D_value,
            )
        )

        if args.tensorboard:
            writer.add_scalar('Loss/train_cls', loss_cls_value, i_iter)
            writer.add_scalar('Loss/train_adv', loss_adv_value, i_iter)
            writer.add_scalar('Loss/train_disc', loss_D_value, i_iter)

        if i_iter % args.iter_save_epoch == 0:
            curr_epoch = i_iter // args.iter_save_epoch
            torch.save(model.state_dict(),os.path.join(args.exp_dir,
                                                       "model_train_epoch_{}.pth").format(curr_epoch))
            torch.save(model_D.state_dict(),os.path.join(args.exp_dir,
                                                         "modelD_train_epoch_{}.pth").format(curr_epoch))

        if i_iter % args.iter_test_epoch == 0:
            curr_accu, curr_loss = run_testing(
                dataloader=testloader,
                model=model,
                criterion=cls_loss,
                logger=test_logger,
                test_iter=i_iter,
                writer=writer,
                args=args,
            )
            if max_test_accu < curr_accu:
                max_test_accu = curr_accu
                max_train_epoch = i_iter // args.iter_test_epoch
                torch.save(model.state_dict(), os.path.join(args.exp_dir,
                                                            "model_train_best.pth"))
                torch.save(model_D.state_dict(), os.path.join(args.exp_dir,
                                                              "modelD_train_best.pth"))

    if args.tensorboard:
        writer.close()

    train_logger.info("Max test accuracy: {:.4f}".format(max_test_accu))
    train_logger.info("Train model is at epoch: {}".format(max_train_epoch))


def run_training_semi(
    trainloader_gt,
    trainloader_nogt,
    trainloader_gt_iter,
    targetloader_nogt_iter,
    testloader,
    model,
    model_D,
    gan_loss,
    cls_loss,
    semi_loss,
    optimizer,
    optimizer_D,
    history_pool_gt,
    history_pool_nogt,
    train_logger,
    test_logger,
    writer,
    args,
):
    gt_label = 1
    nogt_label = 0
    max_test_accu = float("-inf")

    for i_iter in range(args.total_iterations):
        loss_cls_value = 0
        loss_regulization = 0
        loss_adv_value = 0
        loss_semi_value = 0
        loss_D_value = 0

        model.train()
        model_D.train()

        optimizer.zero_grad()
        optimizer_D.zero_grad()

        # if args.adjust_lr:
        #     adjust_learning_rate(
        #         optimizer=optimizer,
        #         learning_rate=args.lr,
        #         i_iter=i_iter,
        #         max_steps=args.total_iterations,
        #         power=0.9,
        #     )
        #
        #     adjust_learning_rate(
        #         optimizer=optimizer_D,
        #         learning_rate=args.lr_D,
        #         i_iter=i_iter,
        #         max_steps=args.total_iterations,
        #         power=0.9,
        #     )

        ## train G ##
        for param in model_D.parameters():
            param.requires_grad = False

        ## train with points w/ GT ##
        try:
            _, batch = next(trainloader_gt_iter)
        except StopIteration:
            trainloader_gt_iter = enumerate(trainloader_gt)
            _, batch = next(trainloader_gt_iter)

        pts, cls = batch
        pts, cls = pts.to(args.device), cls.long().to(args.device)

        pred, global_gt, high_feat = model(pts)
        l_cls = cls_loss(pred, cls)
        loss_cls_value += l_cls.item()
        # global_softmax = F.log_softmax(global_gt, dim=1)
        pred_gt_softmax = F.log_softmax(pred, dim=1)
        # if high_feat is not None:
        #     l_regu = feature_transform_regularizer(high_feat)
        #     loss_regulization += l_regu.item()
        # else:
        #     l_regu = None

        ## train with target ##
        try:
            _, batch = next(targetloader_nogt_iter)
        except StopIteration:
            targetloader_nogt_iter = enumerate(trainloader_nogt)
            _, batch = next(targetloader_nogt_iter)

        pts_nogt = batch
        pts_nogt = pts_nogt.to(args.device)

        pred_nogt, global_nogt, high_feat = model(pts_nogt)
        # global_nogt_softmax = F.log_softmax(global_nogt, dim=1)
        pred_nogt_softmax = F.log_softmax(pred_nogt, dim=1)
        # if high_feat is not None:
        #     l_regu = feature_transform_regularizer(high_feat)
        #     loss_regulization += l_regu.item()
        # else:
        #     l_regu = None

        D_out = model_D(pred_nogt_softmax)
        generated_label = make_D_label(
            input=D_out,
            value=gt_label,
            device=args.device,
            random=False,
        )
        l_adv = gan_loss(D_out, generated_label)
        loss_adv_value += l_adv.item()

        ## semi loss for unlabeled pts
        if (args.semi_start>0) and (i_iter > args.semi_start):
            semi_ignore_mask = (D_out <= args.semi_TH).squeeze(1)
            semi_gt = torch.argmax(pred_nogt.data.cpu(), dim=1)
            semi_gt[semi_ignore_mask] = 255

            semi_ratio = 1.0 - float(semi_ignore_mask.sum().item()) / float(np.prod(semi_ignore_mask.shape))
            if semi_ratio == 0.0:
                loss_semi_value += 0
                l_semi = None
            else:
                semi_gt = torch.LongTensor(semi_gt).to(args.device)
                l_semi = semi_loss(pred_nogt, semi_gt)
                loss_semi_value += l_semi.item()
        else:
            l_semi = None

        # if (l_semi is not None) and (l_regu is None):
        #     loss_CLS_Net = args.lambda_cls * l_cls + \
        #                    args.lambda_adv * l_adv +\
        #                    args.lambda_semi * l_semi
        # elif (l_semi is not None) and (l_regu is not None):
        #     loss_CLS_Net = args.lambda_cls * l_cls + \
        #                    args.lambda_adv * l_adv + \
        #                    args.lambda_semi * l_semi + \
        #                    args.lambda_regu * l_regu
        # else:
        #     loss_CLS_Net = args.lambda_cls * l_cls + \
        #                    args.lambda_adv * l_adv

        if l_semi is not None:
            loss_CLS_Net = args.lambda_cls * l_cls + \
                           args.lambda_adv * l_adv + \
                           args.lambda_semi * l_semi
        else:
            loss_CLS_Net = args.lambda_cls * l_cls + \
                           args.lambda_adv * l_adv

        loss_CLS_Net.backward()

        #loss = args.lambda_cls * l + \
        #       args.lambda_adv * loss_adv
        #loss.backward()

        ## train D ##
        for param in model_D.parameters():
            param.requires_grad = True

        ## train w/ GT ##
        pred_gt_softmax = pred_gt_softmax.detach()
        pool_gt = history_pool_gt.query(pred_gt_softmax)
        D_out = model_D(pool_gt)
        generated_label = make_D_label(
            input=D_out,
            value=gt_label,
            device=args.device,
            random=True,
        )
        loss_D = gan_loss(D_out, generated_label)
        loss_D = loss_D * 0.5
        loss_D.backward()
        loss_D_value += loss_D.item()

        ## train wo GT ##
        pred_nogt_softmax = pred_nogt_softmax.detach()
        pool_nogt = history_pool_nogt.query(pred_nogt_softmax)
        D_out = model_D(pool_nogt)
        generated_label = make_D_label(
            input=D_out,
            value=nogt_label,
            device=args.device,
            random=True,
        )
        loss_D = gan_loss(D_out, generated_label)
        loss_D = loss_D * 0.5
        loss_D.backward()
        loss_D_value += loss_D.item()

        optimizer.step()
        optimizer_D.step()

        train_logger.info('iter = {0:8d}/{1:8d} '
              'loss_cls = {2:.3f} '
              'loss_adv = {3:.3f} '
              # 'loss_regu = {4:.3f} '
              'loss_D = {4:.3f}'.format(
                i_iter, args.total_iterations,
                loss_cls_value,
                loss_adv_value,
                # loss_regulization,
                loss_D_value,
            )
        )

        if args.tensorboard:
            writer.add_scalar('Loss/train_cls', loss_cls_value, i_iter)
            writer.add_scalar('Loss/train_adv', loss_adv_value, i_iter)
            writer.add_scalar('Loss/train_disc', loss_D_value, i_iter)

        if i_iter % args.iter_save_epoch == 0:
            curr_epoch = int(round(i_iter / trainloader_gt.__len__()))
            torch.save(model.state_dict(),os.path.join(args.exp_dir,
                                                       "model_train_epoch_{}.pth").format(curr_epoch))
            torch.save(model_D.state_dict(),os.path.join(args.exp_dir,
                                                         "modelD_train_epoch_{}.pth").format(curr_epoch))

        if i_iter % args.iter_test_epoch == 0:
            curr_accu, curr_loss = run_testing(
                dataloader=testloader,
                model=model,
                criterion=cls_loss,
                logger=test_logger,
                test_iter=i_iter,
                writer=writer,
                args=args,
            )
            if max_test_accu < curr_accu:
                max_test_accu = curr_accu
                max_train_epoch = i_iter // args.iter_test_epoch
                torch.save(model.state_dict(), os.path.join(args.exp_dir,
                                                            "model_train_best.pth"))
                torch.save(model_D.state_dict(), os.path.join(args.exp_dir,
                                                              "modelD_train_best.pth"))

    if args.tensorboard:
        writer.close()

    train_logger.info("Max test accuracy: {:.4f}".format(max_test_accu))
    train_logger.info("Train model is at epoch: {}".format(max_train_epoch))

def run_training_seg(
    trainloader_gt,
    trainloader_nogt,
    trainloader_gt_iter,
    targetloader_nogt_iter,
    testloader,
    testdataset,
    model,
    model_D,
    gan_loss,
    seg_loss,
    optimizer,
    optimizer_D,
    history_pool_gt,
    history_pool_nogt,
    train_logger,
    test_logger,
    writer,
    args,
):
    gt_label = 1
    nogt_label = 0
    max_seg_accu = float("-inf")

    for i_iter in range(args.total_iterations):
        loss_seg_value = 0
        loss_adv_value = 0
        loss_regulization = 0
        loss_D_value = 0

        model.train()
        model_D.train()

        optimizer.zero_grad()
        optimizer_D.zero_grad()

        # adjust_learning_rate(
        #     optimizer=optimizer_SS,
        #     learning_rate=args.lr_SS,
        #     i_iter=i_iter,
        #     max_steps=args.total_iterations,
        #     power=0.9,
        # )
        #
        # adjust_learning_rate(
        #     optimizer=optimizer_D,
        #     learning_rate=args.lr_D,
        #     i_iter=i_iter,
        #     max_steps=args.total_iterations,
        #     power=0.9,
        # )

        ## train G ##
        for param in model_D.parameters():
            param.requires_grad = False

        ## train with points w/ GT ##
        try:
            _, batch = next(trainloader_gt_iter)
        except StopIteration:
            trainloader_gt_iter = enumerate(trainloader_gt)
            _, batch = next(trainloader_gt_iter)

        pts, cls, seg = batch
        pts, cls, seg = pts.to(args.device), cls.to(args.device), seg.long().to(args.device)

        pred, global_gt, input_feat, high_feat = model(pts, cls)
        l_seg = seg_loss(pred, seg)
        loss_seg_value += l_seg.item()
        pred_gt_softmax = F.softmax(pred, dim=1)
        if high_feat is not None:
            l_regu = feature_transform_regularizer(high_feat)
            loss_regulization += l_regu.item()
        else:
            l_regu = None

        ## train with target ##
        try:
            _, batch = next(targetloader_nogt_iter)
        except StopIteration:
            targetloader_nogt_iter = enumerate(trainloader_nogt)
            _, batch = next(targetloader_nogt_iter)

        pts_nogt, cls_nogt = batch
        pts_nogt, cls_nogt = pts_nogt.to(args.device), cls_nogt.to(args.device)

        pred_nogt, global_nogt, input_feat, high_feat = model(pts_nogt, cls_nogt)
        pred_nogt_softmax = F.log_softmax(pred_nogt, dim=1)
        if high_feat is not None:
            l_regu = feature_transform_regularizer(high_feat)
            loss_regulization += l_regu.item()
        else:
            l_regu = None

        D_out = model_D(pred_nogt_softmax)  #Bx1xC
        generated_label = make_D_label(
            input=D_out,
            value=gt_label,
            device=args.device,
            random=False,
        )

        loss_adv = gan_loss(D_out, generated_label)
        loss_adv_value += loss_adv.item()

        if l_regu is None:
            loss = args.lambda_seg * l_seg + \
                   args.lambda_adv * loss_adv
        else:
            loss = args.lambda_seg * l_seg + \
                   args.lambda_adv * loss_adv + \
                   args.lambda_regu * l_regu
        loss.backward()

        ## train D ##
        for param in model_D.parameters():
            param.requires_grad = True

        ## train w/ GT ##
        pred_gt_softmax = pred_gt_softmax.detach()
        pool_gt = history_pool_gt.query(pred_gt_softmax)
        D_out = model_D(pool_gt)
        generated_label = make_D_label(
            input=D_out,
            value=gt_label,
            device=args.device,
            random=True,
        )
        loss_D = gan_loss(D_out, generated_label)
        loss_D = loss_D * 0.5
        loss_D.backward()
        loss_D_value += loss_D.item()

        ## train wo GT ##
        pred_nogt_softmax = pred_nogt_softmax.detach()
        pool_nogt = history_pool_nogt.query(pred_nogt_softmax)
        D_out = model_D(pool_nogt)
        generated_label = make_D_label(
            input=D_out,
            value=nogt_label,
            device=args.device,
            random=True,
        )
        loss_D = gan_loss(D_out, generated_label)
        loss_D = loss_D * 0.5
        loss_D.backward()
        loss_D_value += loss_D.item()

        optimizer.step()
        optimizer_D.step()

        train_logger.info('iter = {0:8d}/{1:8d} '
              'loss_seg = {2:.3f} '
              'loss_adv = {3:.3f} '
              'loss regu = {4:.3f} '
              'loss_D = {5:.3f}'.format(
                i_iter, args.total_iterations,
                loss_seg_value,
                loss_adv_value,
                loss_regulization,
                loss_D_value,
            )
        )

        if args.tensorboard:
            writer.add_scalar('Loss/train_seg', loss_seg_value, i_iter)
            writer.add_scalar('Loss/train_adv', loss_adv_value, i_iter)
            writer.add_scalar('Loss/train_disc', loss_D_value, i_iter)

        if i_iter % args.iter_test_epoch == 0:
            curr_epoch = i_iter // args.iter_test_epoch
            torch.save(model.state_dict(), os.path.join(args.exp_dir,
                                                       "model_train_epoch_{}.pth").format(curr_epoch))
            torch.save(model_D.state_dict(), os.path.join(args.exp_dir,
                                                         "modelD_train_epoch_{}.pth").format(curr_epoch))

            curr_accu, curr_loss = run_testing_seg(
                dataloader=testloader,
                dataset=testdataset,
                model=model,
                criterion=seg_loss,
                logger=test_logger,
                test_iter=i_iter,
                writer=writer,
                args=args,
            )

            if max_seg_accu < curr_accu:
                max_seg_accu = curr_accu
                max_train_epoch = i_iter // args.iter_test_epoch
                torch.save(model.state_dict(), os.path.join(args.exp_dir,
                                                            "model_train_best.pth"))
                torch.save(model_D.state_dict(), os.path.join(args.exp_dir,
                                                              "modelD_train_best.pth"))

    test_logger.info("Max test accuracy: {:.4f} ".format(max_seg_accu))
    test_logger.info("Max train epoch: {} ".format(max_train_epoch))

    if args.tensorboard:
        writer.close()


def run_training_seg_semi(
    trainloader_gt,
    trainloader_nogt,
    trainloader_gt_iter,
    targetloader_nogt_iter,
    testloader,
    testdataset,
    model,
    model_D,
    gan_loss,
    seg_loss,
    semi_loss,
    optimizer,
    optimizer_D,
    history_pool_gt,
    history_pool_nogt,
    train_logger,
    test_logger,
    writer,
    args,
):
    gt_label = 1
    nogt_label = 0
    max_seg_accu = float("-inf")

    for i_iter in range(args.total_iterations):
        loss_seg_value = 0
        loss_adv_value = 0
        loss_semi_value = 0
        loss_regulization = 0
        loss_D_value = 0

        model.train()
        model_D.train()

        optimizer.zero_grad()
        optimizer_D.zero_grad()

        # adjust_learning_rate(
        #     optimizer=optimizer_SS,
        #     learning_rate=args.lr_SS,
        #     i_iter=i_iter,
        #     max_steps=args.total_iterations,
        #     power=0.9,
        # )
        #
        # adjust_learning_rate(
        #     optimizer=optimizer_D,
        #     learning_rate=args.lr_D,
        #     i_iter=i_iter,
        #     max_steps=args.total_iterations,
        #     power=0.9,
        # )

        ## train G ##
        for param in model_D.parameters():
            param.requires_grad = False

        ## train with points w/ GT ##
        try:
            _, batch = next(trainloader_gt_iter)
        except StopIteration:
            trainloader_gt_iter = enumerate(trainloader_gt)
            _, batch = next(trainloader_gt_iter)

        pts, cls, seg = batch
        pts, cls, seg = pts.to(args.device), cls.to(args.device), seg.long().to(args.device)

        pred, global_gt, input_feat, high_feat = model(pts, cls)
        l_seg = seg_loss(pred, seg)
        loss_seg_value += l_seg.item()
        pred_gt_softmax = F.softmax(pred, dim=1)
        if high_feat is not None:
            l_regu = feature_transform_regularizer(high_feat)
            loss_regulization += l_regu.item()
        else:
            l_regu = None

        ## train with target ##
        try:
            _, batch = next(targetloader_nogt_iter)
        except StopIteration:
            targetloader_nogt_iter = enumerate(trainloader_nogt)
            _, batch = next(targetloader_nogt_iter)

        pts_nogt, cls_nogt = batch
        pts_nogt, cls_nogt = pts_nogt.to(args.device), cls_nogt.to(args.device)

        pred_nogt, global_nogt, input_feat, high_feat = model(pts_nogt, cls_nogt)
        pred_nogt_softmax = F.log_softmax(pred_nogt, dim=1)
        if high_feat is not None:
            l_regu = feature_transform_regularizer(high_feat)
            loss_regulization += l_regu.item()
        else:
            l_regu = None

        D_out = model_D(pred_nogt_softmax)  #Bx1xC
        generated_label = make_D_label(
            input=D_out,
            value=gt_label,
            device=args.device,
            random=False,
        )

        l_adv = gan_loss(D_out, generated_label)
        loss_adv_value += l_adv.item()

        # if l_regu is None:
        #     loss = args.lambda_seg * l_seg + \
        #            args.lambda_adv * loss_adv
        # else:
        #     loss = args.lambda_seg * l_seg + \
        #            args.lambda_adv * loss_adv + \
        #            args.lambda_regu * l_regu
        # loss.backward()

        ## semi loss for unlabeled pts ##
        if (args.semi_start > 0) and (i_iter > args.semi_start):
            semi_ignore_mask = (D_out <= args.semi_TH).squeeze(1)
            semi_gt = torch.argmax(pred_nogt.data.cpu(), dim=1)
            semi_gt[semi_ignore_mask] = 255

            semi_ratio = 1.0 - float(semi_ignore_mask.sum().item()) / float(np.prod(semi_ignore_mask.shape))
            if semi_ratio == 0.0:
                loss_semi_value += 0
                l_semi = None
            else:
                semi_gt = torch.LongTensor(semi_gt).to(args.device)
                l_semi = semi_loss(pred_nogt, semi_gt)
                loss_semi_value += l_semi.item()
        else:
            l_semi = None

        if (l_semi is not None) and (l_regu is None):
            loss_SEG_Net = args.lambda_seg * l_seg + \
                           args.lambda_adv * l_adv + \
                           args.lambda_semi * l_semi
        elif (l_semi is not None) and (l_regu is not None):
            loss_SEG_Net = args.lambda_seg * l_seg + \
                           args.lambda_adv * l_adv + \
                           args.lambda_semi * l_semi + \
                           args.lambda_regu * l_regu
        else:
            loss_SEG_Net = args.lambda_seg * l_seg + \
                           args.lambda_adv * l_adv

        loss_SEG_Net.backward()
        optimizer.step()

        ## train D ##
        for param in model_D.parameters():
            param.requires_grad = True

        ## train w/ GT ##
        pred_gt_softmax = pred_gt_softmax.detach()
        pool_gt = history_pool_gt.query(pred_gt_softmax)
        D_out = model_D(pool_gt)
        generated_label = make_D_label(
            input=D_out,
            value=gt_label,
            device=args.device,
            random=True,
        )
        loss_D = gan_loss(D_out, generated_label)
        loss_D = loss_D * 0.5
        loss_D.backward()
        loss_D_value += loss_D.item()

        ## train wo GT ##
        pred_nogt_softmax = pred_nogt_softmax.detach()
        pool_nogt = history_pool_nogt.query(pred_nogt_softmax)
        D_out = model_D(pool_nogt)
        generated_label = make_D_label(
            input=D_out,
            value=nogt_label,
            device=args.device,
            random=True,
        )
        loss_D = gan_loss(D_out, generated_label)
        loss_D = loss_D * 0.5
        loss_D_value += loss_D.item()
        loss_D.backward()

        optimizer_D.step()

        train_logger.info('iter = {0:8d}/{1:8d} '
              'loss_seg = {2:.3f} '
              'loss_adv = {3:.3f} '
              'loss semi = {4:.3f} '
              'loss regu = {5:.3f} '
              'loss_D = {6:.3f}'.format(
                i_iter, args.total_iterations,
                loss_seg_value,
                loss_adv_value,
                loss_semi_value,
                loss_regulization,
                loss_D_value,
            )
        )

        if args.tensorboard:
            writer.add_scalar('Loss/train_seg', loss_seg_value, i_iter)
            writer.add_scalar('Loss/train_adv', loss_adv_value, i_iter)
            writer.add_scalar('Loss/train_disc', loss_D_value, i_iter)

        if i_iter % args.iter_test_epoch == 0:
            curr_epoch = i_iter // args.iter_test_epoch
            torch.save(model.state_dict(), os.path.join(args.exp_dir,
                                                       "model_train_epoch_{}.pth").format(curr_epoch))
            torch.save(model_D.state_dict(), os.path.join(args.exp_dir,
                                                         "modelD_train_epoch_{}.pth").format(curr_epoch))

            curr_accu, curr_loss = run_testing_seg(
                dataloader=testloader,
                dataset=testdataset,
                model=model,
                criterion=seg_loss,
                logger=test_logger,
                test_iter=i_iter,
                writer=writer,
                args=args,
            )

            if max_seg_accu < curr_accu:
                max_seg_accu = curr_accu
                max_train_epoch = i_iter // args.iter_test_epoch

    test_logger.info("Max test accuracy: {:.4f} ".format(max_seg_accu))
    test_logger.info("Max train epoch: {} ".format(max_train_epoch))

    if args.tensorboard:
        writer.close()