import os
import sys

import torch
from torch.nn import init

from models.discriminator import DeepConvDiscNet, PointwiseDiscNet
from models.discriminator import BaseDiscNet, ShapeDiscNet, PointDiscNet
from models.pointnet import PointNetCls, PointNetSeg

def init_net(net, device, init_type, init_gain=1.0):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    net.to(device)
    if init_type is None:
        return net
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type, init_gain=1.0):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    if init_type is None:
        return net
    net.apply(init_func)  # apply the initialization function <init_func>

def load_models(mode, device, args):
    """

    :param mode: "cls" or "disc"
    :param args:
    :return:
    """

    if mode == "cls":
        model = PointNetCls(k=40, feature_transform=False) # False
        model = model.to(device)
        try:
            if args.checkpoint:
                print('===============================')
                print("Loading pretrained cls model ...")
                print('===============================')
                model.load_state_dict(torch.load(args.checkpoint))
        except Exception as e:
            print(e)
            sys.exit(0)
    elif mode == "seg":
        model = PointNetSeg(NUM_SEG_CLASSES=50)
        model = model.to(device)
        try:
            if args.checkpoint:
                print('===============================')
                print("Loading pretrained cls model ...")
                print('===============================')
                model.load_state_dict(torch.load(args.checkpoint))
        except Exception as e:
            print(e)
            sys.exit(0)
    elif mode == "disc":
        model = DeepConvDiscNet(input_dim=args.disc_indim, output_dim=1)  # input_dim=args.input_pts, output_dim=1)
        model = init_net(model, device, init_type=args.init_disc)
        #if args.checkpoint_disc:
        #    print('===============================')
        #    print("Loading pretrained discriminator model ...")
        #    print('===============================')
        #    model.load_state_dict(torch.load(args.checkpoint_disc))
        # model = init_net(model, device, init_type=args.init_disc)
    elif mode == "disc_seg":
        model = PointwiseDiscNet(input_pts=args.input_pts,input_dim=args.disc_indim)
        model = init_net(model, device, init_type=args.init_disc)

    elif mode == "disc_dual":
        shared_disc = BaseDiscNet(input_pts=args.input_pts,input_dim=args.disc_indim, output_dim=256)
        shapeDisc = ShapeDiscNet(shared_output_dim=256, num_shapes=16)
        pointDisc = PointDiscNet(shared_output_dim=256, input_pts=args.input_pts)
        # model_point = SharedPointDiscNet(
        #     input_pts=args.input_pts,
        #     input_dim=args.disc_indim,
        #     shared_output_dim=128)
        # model_shape = SharedShapeDiscNet(
        #     input_pts=args.input_pts,
        #     input_dim=args.disc_indim,
        #     shared_output_dim=128,
        #     num_shapes=16,
        # )
        shared_disc = init_net(shared_disc, device, init_type=args.init_disc)
        shapeDisc = init_net(shapeDisc, device, init_type=args.init_disc)
        pointDisc = init_net(pointDisc, device, init_type=args.init_disc)
        return shared_disc, shapeDisc, pointDisc
    else:
        raise ValueError("Invalid mode {}!".format(mode))

    return model

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate_SS(optimizer, learning_rate, i_iter, max_steps, power):
    lr = lr_poly(learning_rate, i_iter, max_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, learning_rate, i_iter, max_steps, power):
    lr = lr_poly(learning_rate, i_iter, max_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
