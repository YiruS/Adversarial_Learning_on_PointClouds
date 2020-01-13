from __future__ import print_function

import os
import sys

import pickle
import argparse
import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from dataset.shapeNetData import ShapeNetDataset, ShapeNetGTDataset

from models.pointnet import PointNetSeg
from models.discriminator import ConvDiscNet

from utils.loss import loss_calc, loss_bce
from utils.utils_train import fastprint, make_D_label
from utils.utils_train import lr_poly, adjust_learning_rate, adjust_learning_rate_D, one_hot
from utils.utils_train import create_dataset, create_GT_dataset, create_dataloader, create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="PointNet-Semi Network")
parser.add_argument("--model", dest="model", type=str, default='PointNetSeg',
                    help="available options : PointNet")
parser.add_argument("--batch-size", dest="batch_size", type=int, default=16,
                    help="Number of images sent to the network in one step.")
parser.add_argument("--iter-size", dest="iter_size", type=int, default=1,
                    help="Accumulate gradients for ITER_SIZE iterations.")
parser.add_argument("--num-workers", dest="num_workers", type=int, default=6,
                    help="number of workers for multithread dataloading.")
parser.add_argument("--partial-data", dest="partial_data", type=float, default=0.25,
                    help="The index of the label to ignore during the training.")
parser.add_argument("--partial-id", dest="partial_id", type=str, default=None,
                    help="restore partial id list")
parser.add_argument("--is-training", action="store_true",
                    help="Whether to updates the running means and variances during the training.")
parser.add_argument("--learning-rate", dest="lr_G", type=float, default=1e-4,
                    help="Base learning rate for training with polynomial decay.")
parser.add_argument("--learning-rate-D", dest="lr_D", type=float, default=1e-4,
                    help="Base learning rate for discriminator.")
parser.add_argument("--lambda-adv", dest="lambda_adv", type=float, default=0.01,
                    help="lambda_adv for adversarial training.")
parser.add_argument("--lambda-semi", dest="lambda_semi", type=float, default=0.1,
                    help="lambda_semi for adversarial training.")
parser.add_argument("--lambda-semi-adv", dest="lambda_semi_adv", type=float, default=0.001,
                    help="lambda_semi for adversarial training.")
parser.add_argument("--mask-T", dest="mask_T", type=float, default=0.1,
                    help="mask T for semi adversarial training.")
parser.add_argument("--semi-start", dest="semi_start", type=int, default=10,
                    help="start semi learning after # iterations (10 epochs by default)")
parser.add_argument("--semi-start-adv", dest="semi_start_adv", type=int, default=20,
                    help="start semi learning after # iterations (20 epochs by default)")
parser.add_argument("--D-remain", dest="D_remain", type=bool, default=True,
                    help="Whether to train D with unlabeled data")
parser.add_argument("--momentum", dest="momentum", type=float, default=0.9,
                    help="Momentum component of the optimiser.")
parser.add_argument("--not-restore-last", action="store_true",
                    help="Whether to not restore last (FC) layers.")
parser.add_argument("--num-instance-classes", dest="num_instance_classes", type=int, default=16,
                    help="Number of classes to predict (including background).")
parser.add_argument("--num-seg-classes", dest="num_seg_classes", type=int, default=50,
                    help="Number of classes to predict (including background).")
parser.add_argument("--num-epochs", dest="num_epoch", type=int, default=200,
                    help="Number of training steps.")
parser.add_argument("--power", dest="power", type=float, default=0.9,
                    help="Decay parameter to compute the learning rate.")
parser.add_argument("--random-mirror", action="store_true",
                    help="Whether to randomly mirror the inputs during the training.")
parser.add_argument("--random-scale", action="store_true",
                    help="Whether to randomly scale the inputs during the training.")
parser.add_argument("--restore-from-D", dest="restore_from_D", type=str, default=None,
                    help="Where restore model parameters from.")
parser.add_argument("--save-pred-every", dest="save_pred_every", type=int, default=2,
                    help="Save summaries and checkpoint every often.")
parser.add_argument("--snapshot-dir", dest="snapshot_dir", type=str, default="./snapshots/",
                    help="Where to save snapshots of the model.")
parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0005,
                    help="Regularisation parameter for L2-loss.")
parser.add_argument("--gpu", dest="gpu", type=int, default=0,
                    help="choose gpu device.")
parser.add_argument("--num-pts", dest="num_pts", type=int, default=2048,
                    help="#points in each instance")
parser.add_argument("--noise", action="store_true", dest="noise", default=False,
                   help="add noise in data augmentation")
parser.add_argument("--rotate", action="store_true", dest="rotate", default=False,
                   help="rotate shape in data augmentation")
parser.add_argument("--shuffle", action="store_true", dest="shuffle", default=False,
                   help="shuffle data")

opts = parser.parse_args()
print(opts)

MODEL = opts.model
BATCH_SIZE = opts.batch_size
ITER_SIZE = opts.iter_size
NUM_WORKERS = opts.num_workers
NUM_PTS = opts.num_pts

LR_G = opts.lr_G
LR_D = opts.lr_D
MOMENTUM = opts.momentum
NUM_INST_CLASSES = opts.num_instance_classes
NUM_SEG_CLASSES = opts.num_seg_classes
NUM_EPOCHS = opts.num_epoch # 20000
POWER = opts.power

IS_SHUFFLE = opts.shuffle
IS_ROTATE = opts.rotate
IS_NOISE = opts.noise
# RANDOM_SEED = 1234

SAVE_PRED_EVERY = opts.save_pred_every # every 2 epoches, batch size 16
SNAPSHOT_DIR = opts.snapshot_dir
WEIGHT_DECAY = opts.weight_decay

LEARNING_RATE_D = opts.lr_D
LAMBDA_ADV = opts.lambda_adv

PARTIAL_DATA = opts.partial_data

SEMI_START = opts.semi_start #10 epochs
LAMBDA_SEMI = opts.lambda_semi
MASK_T = opts.mask_T

LAMBDA_SEMI_ADV = opts.lambda_semi_adv
SEMI_START_ADV = opts.semi_start_adv # 20 epochs
D_REMAIN = opts.D_remain

GT_LABEL = 1
PRED_LABEL = 0

torch.backends.cudnn.enabled = True


def train(opts):
    if not os.path.exists(opts.snapshot_dir):
        os.makedirs(opts.snapshot_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_G = create_model(type="generator", num_seg_classes=opts.num_seg_classes)
    model_D = create_model(type="discriminator", num_seg_classes=opts.num_seg_classes)

    model_G.to(device)
    model_G.train()

    model_D.to(device)
    model_D.train()

    train_dataset = create_dataset(
        num_inst_classes = NUM_INST_CLASSES,
        num_pts = NUM_PTS,
        mode = "train",
        is_noise = IS_NOISE,
        is_rotate = IS_ROTATE,
    )
    train_dataset_size = len(train_dataset)
    print("#Total train: {:6d}".format(train_dataset_size))

    if opts.partial_data is None:
        trainloader = create_dataloader(
            dataset = train_dataset,
            batch_size = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            shuffle = IS_SHUFFLE,
            pin_memory = True,
        )
    else:
        partial_size = int(opts.partial_data * train_dataset_size)

        if opts.partial_id is not None:
            train_ids = pickle.load(open(opts.partial_id))
            print('loading train ids from {}'.format(opts.partial_id))
        else:
            train_ids = list(range(train_dataset_size))
            np.random.shuffle(train_ids)

        pickle.dump(train_ids, open(os.path.join(opts.snapshot_dir, 'train_id.pkl'), 'wb'))

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_ids[:partial_size])
        train_remain_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_ids[partial_size:])

        trainloader = create_dataloader(
            dataset = train_dataset,
            batch_size = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            shuffle = IS_SHUFFLE,
            pin_memory = True,
            sampler = train_sampler,
        )
        trainloader_remain = create_dataloader(
            dataset = train_dataset,
            batch_size = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            shuffle = IS_SHUFFLE,
            pin_memory = True,
            sampler = train_remain_sampler,
        )

    # optimizer for segmentation network
    optimizer_G = optim.Adam(model_G.parameters(), lr = opts.lr_G, betas = (0.9, 0.999))
    optimizer_G.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr = opts.lr_G, betas = (0.9, 0.999))
    optimizer_D.zero_grad()

    for epoch in np.arange(NUM_EPOCHS):
        print("========== EPOCH {:6d} ==========".format(epoch+1))
        if epoch >= 0 and epoch <= 9: # only train generator
            loss_ce_value, loss_adv_value = train_generator(
                model_G,
                model_D,
                trainloader,
                optimizer_G,
                train_dataset_size,
                device
            )
            fastprint('[%d/%d] CE loss: %.3f, ADV loss: %.3f' %
                      (epoch,
                       NUM_EPOCHS,
                       loss_ce_value,
                       loss_adv_value)
                      )
        elif epoch >= 10 and epoch <=19: # only train discriminator
            loss_D_value = train_discriminator(
                model_G,
                model_D,
                trainloader,
                optimizer_D,
                train_dataset_size,
                device
            )
            fastprint('[%d/%d] Discriminator loss: %.3f' %
                      (epoch,
                       NUM_EPOCHS,
                       loss_D_value)
                      )
        else: # start unlabeled data
            print("==== SEMI ====")
            pred_remain, ignore_mask_remain, loss_semi_adv_value, loss_semi_value = train_semi(
                model_G,
                model_D,
                trainloader_remain,
                optimizer_G,
                train_dataset_size,
                device
            )
            print("==== SEMI w/ Generator ====")
            loss_ce_value, loss_adv_value = train_generator(
                model_G,
                model_D,
                trainloader,
                optimizer_G,
                train_dataset_size,
                device
            )
            print("==== SEMI w/ Discriminator ====")
            loss_D_value = train_discriminator(
                model_G,
                model_D,
                trainloader,
                optimizer_D,
                device,
                train_dataset_size,
                pred_remain,
                ignore_mask_remain
            )
            fastprint('[%d/%d] Semi_adv: %.3f, Semi: %.3f, Total Semi: %.3f, '
                      'CE: %.3f, ADV: %.3f, D: %.3f' %
                      (epoch,
                       NUM_EPOCHS,
                       loss_semi_adv_value,
                       loss_semi_value,
                       loss_semi_adv_value+loss_semi_value,
                       loss_ce_value,
                       loss_adv_value,
                       loss_D_value)
                      )

        if epoch > 0 and epoch%2==0:
            print('taking snapshot ...')
            torch.save(model_G.state_dict(), os.path.join(SNAPSHOT_DIR, 'VOC_' + str(epoch) + '_G.pth'))
            torch.save(model_D.state_dict(), os.path.join(SNAPSHOT_DIR, 'VOC_' + str(epoch) + '_D.pth'))

def train_generator(
        model_G,
        model_D,
        trainloader,
        optimizer_G,
        train_dataset_size,
        device
):
    """
    Train generator (segmentation network), including loss_ce and loss_adv with GT.
    :return: loss_ce, loss_adv
    """
    loss_ce_value = []
    loss_adv_value = []
    NUM_BATCHES = np.floor(train_dataset_size/BATCH_SIZE)

    for i, mini_batch in tqdm.tqdm(enumerate(trainloader), total=NUM_BATCHES):
        # don't accumulate grads in D
        for param in model_G.parameters():
            param.requires_grad = True
        for param in model_D.parameters():
            param.requires_grad = False

        optimizer_G.zero_grad()

        points, cls_gt, seg_gt = mini_batch
        points, cls_gt, seg_gt = Variable(points).float(), \
                                 Variable(cls_gt).float(), \
                                 Variable(seg_gt).type(torch.LongTensor)
        points, cls_gt, seg_gt = points.to(device), \
                                 cls_gt.to(device), \
                                 seg_gt.to(device)
        pred = model_G(points, cls_gt)
        # loss_ce
        loss_ce = loss_calc(pred, seg_gt, device, mask=False)
        # loss_adv
        D_out = model_D(F.softmax(pred, dim=2))
        ignore_mask = np.zeros(seg_gt.shape).astype(np.bool)
        loss_adv = loss_bce(D_out, make_D_label(GT_LABEL, ignore_mask, device), device)
        loss_seg = loss_ce + LAMBDA_ADV * loss_adv
        loss_seg.backward()
        optimizer_G.step()

        loss_ce_value.append(loss_ce.item())
        loss_adv_value.append(loss_adv.item())

    return np.average(loss_ce_value), np.average(loss_adv_value)

def train_discriminator(
        model_G,
        model_D,
        trainloader,
        optimizer_D,
        device,
        train_dataset_size,
        pred_remain = None,
        ignore_mask_remain = None,
):
    """
    Train discriminator
    :return: L_D (trained w/ GT and Pred)
    """
    loss_D_value = []
    NUM_BATCHES = np.floor(train_dataset_size / BATCH_SIZE)
    for i, mini_batch in tqdm.tqdm(enumerate(trainloader), total=NUM_BATCHES):
        # don't accumulate grads in G
        for param in model_G.parameters():
            param.requires_grad = False
        for param in model_D.parameters():
            param.requires_grad = True

        optimizer_D.zero_grad()

        points, cls_gt, seg_gt = mini_batch
        points, cls_gt, seg_gt = Variable(points).float(), \
                                 Variable(cls_gt).float(), \
                                 Variable(seg_gt).type(torch.LongTensor)
        points, cls_gt, seg_gt = points.to(device), \
                                 cls_gt.to(device), \
                                 seg_gt.to(device)

        ignore_mask_gt = np.zeros(seg_gt.shape).astype(np.bool)
        D_gt_v = Variable(one_hot(seg_gt, NUM_SEG_CLASSES)).float().to(device)
        D_out = model_D(D_gt_v)
        loss_D_gt = loss_bce(D_out, make_D_label(
            GT_LABEL,
            ignore_mask_gt,
            device),
            device
        )

        ignore_mask = np.zeros(seg_gt.shape).astype(np.bool)
        pred = model_G(points, cls_gt)
        pred = pred.detach()

        if pred_remain is not None:
            pred = torch.cat((pred, pred_remain), 0)
            ignore_mask = np.concatenate((ignore_mask, ignore_mask_remain), axis=0)

        D_out = model_D(F.softmax(pred, dim=2))
        loss_D_pred = loss_bce(D_out, make_D_label(
            PRED_LABEL,
            ignore_mask,
            device),
            device
        )
        loss_D = loss_D_gt + loss_D_pred
        loss_D.backward()
        optimizer_D.step()

        loss_D_value.append(loss_D.item())

    return np.average(loss_D_value)

def train_semi(
        model_G,
        model_D,
        trainloader_remain,
        optimizer_G,
        train_dataset_size,
        device
):
    """
    Train when GT is NOT available, needs to train loss_semi_adv and loss_semi (for generator).
    :return: loss_semi_value = loss_semi_adv + loss_semi
    """
    loss_semi_adv_value = []
    loss_semi_value = []
    NUM_BATCHES = np.floor(train_dataset_size / BATCH_SIZE)
    for i, mini_batch in tqdm.tqdm(enumerate(trainloader_remain), total = NUM_BATCHES):
        # don't accumulate grads in D
        for param in model_G.parameters():
            param.requires_grad = True
        for param in model_D.parameters():
            param.requires_grad = False

        optimizer_G.zero_grad()

        # only access to points
        points, cls, _ = mini_batch
        points, cls = Variable(points).float(), Variable(cls).float()
        points, cls = points.to(device), cls.to(device)

        pred = model_G(points, cls)
        pred_remain = pred.detach()

        D_out = model_D(F.softmax(pred, dim=2))
        D_out_sigmoid = torch.sigmoid(D_out).data.cpu().numpy()  # BxN
        ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)  # Bx2048

        ### semi_adv ###
        loss_semi_adv = LAMBDA_SEMI_ADV * loss_bce(D_out, make_D_label(
            GT_LABEL,
            ignore_mask_remain,
            device),
            device
        )

        ### semi ###
        semi_ignore_mask = (D_out_sigmoid < MASK_T)
        semi_gt = pred.data.cpu().numpy().argmax(axis=2)
        semi_gt[semi_ignore_mask] = 999

        semi_ratio = 1.0 - float(semi_ignore_mask.sum()) / semi_ignore_mask.size
        print('semi ratio: {:.4f}'.format(semi_ratio))

        if semi_ratio == 0.0:
            raise ValueError("Semi ratio == 0!")
        else:
            semi_gt = torch.FloatTensor(semi_gt)
            loss_semi = LAMBDA_SEMI * loss_calc(pred, semi_gt, device, mask=True)
            loss_semi_value.append(loss_semi.item())
            loss_semi += loss_semi_adv
            loss_semi.backward()
            optimizer_G.step()
            loss_semi_adv_value.append(loss_semi_adv.item())

    return pred_remain, ignore_mask_remain, np.average(loss_semi_adv_value), np.average(loss_semi_value)

if __name__ == '__main__':
    train(opts)
