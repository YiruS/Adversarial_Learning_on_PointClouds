import os
import sys

import numpy as np
import logging

import torch

def make_logger(filename, args):
    logger = logging.getLogger()
    file_log_handler = logging.FileHandler(os.path.join(args.exp_dir, filename))
    logger.addHandler(file_log_handler)
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stderr_log_handler)
    logger.setLevel('INFO')
    formatter = logging.Formatter()
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)
    logger.info(args)
    return logger

def make_D_label(input, value, device, random=False):
    if random:
        if value == 0:
            lower, upper = 0, 0.305
        elif value == 1:
            lower, upper = 0.7, 1.05
        D_label = torch.FloatTensor(input.data.size()).uniform_(lower, upper).to(device)
    else:
        D_label = torch.FloatTensor(input.data.size()).fill_(value).to(device)
    return D_label

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, learning_rate, i_iter, max_steps, power):
    lr = lr_poly(learning_rate, i_iter, max_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def get_semi_gt(D_outs, pred, pred_softmax, threshold, device):
    count = 0
    for i in range(D_outs.size(0)):
        if D_outs[i] > threshold:
            count += 1

    if count > 0:
        # print('> threshold: ', count, '/', D_outs.shape[0])
        pred_sel = torch.Tensor(count, pred.size(1), pred.size(2), pred.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0
        for j in range(D_outs.size(0)):
            if D_outs[j] > threshold:
                pred_sel[num_sel] = pred[j]
                semi_gt = torch.argmax(pred_softmax[j].data.cpu(), dim=0, keepdim=False)
                label_sel[num_sel] = semi_gt
                num_sel += 1
        return pred_sel.to(device), label_sel.long().to(device), count
    else:
        return 0, 0, count

def one_hot(label, device, num_classes=4):
    label = label.data.cpu().numpy()
    one_hot = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[:,i,...] = (label == i)
    return torch.FloatTensor(one_hot).to(device)