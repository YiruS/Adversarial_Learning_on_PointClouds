"""
Code from https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(
            torch.from_numpy(
                np.eye(self.k).flatten().astype(np.float32))
        ).view(1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        # self.stn = STN3d()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

        #self.conv1 = torch.nn.Conv1d(3, 64, 1)
        #self.conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.conv3 = torch.nn.Conv1d(128, 128, 1)
        #self.conv4 = torch.nn.Conv1d(128, 128, 1)
        #self.bn1 = nn.BatchNorm1d(64)
        #self.bn2 = nn.BatchNorm1d(128)
        #self.bn3 = nn.BatchNorm1d(128)
        #self.bn4 = nn.BatchNorm1d(128)
        #self.global_feat = global_feat
        #self.feature_transform = feature_transform
        #if self.feature_transform:
        #    self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2] # BxCxN
        # trans = self.stn(x) # BxNxN
        # x = x.transpose(2, 1) # BxNxC
        # x = torch.bmm(x, trans)
        # x = x.transpose(2, 1) # BxCxN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=True)[0] # BxCxN -> BxCx1
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans_feat
            # return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans_feat
            # return torch.cat([x, pointfeat], 1), trans, trans_feat

# class PointNetfeat_SEG(nn.Module):
#     def __init__(self, global_feat=True, feature_transform=False):
#         super(PointNetfeat_SEG, self).__init__()
#         self.stn = STN3d()
#
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 128, 1)
#         self.conv4 = torch.nn.Conv1d(128, 128, 1)
#         self.conv5 = torch.nn.Conv1d(128, 512, 1)
#         self.conv6 = torch.nn.Conv1d(512, 2048, 1)
#
#         self.global_feat = global_feat
#         self.feature_transform = feature_transform
#         if self.feature_transform:
#             self.fstn = STNkd(k=64)
#
#         #self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         #self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         #self.conv3 = torch.nn.Conv1d(128, 128, 1)
#         #self.conv4 = torch.nn.Conv1d(128, 128, 1)
#         #self.bn1 = nn.BatchNorm1d(64)
#         #self.bn2 = nn.BatchNorm1d(128)
#         #self.bn3 = nn.BatchNorm1d(128)
#         #self.bn4 = nn.BatchNorm1d(128)
#         #self.global_feat = global_feat
#         #self.feature_transform = feature_transform
#         #if self.feature_transform:
#         #    self.fstn = STNkd(k=64)
#
#     def forward(self, x, cls):
#         n_pts = x.size()[2] # BxCxN
#         trans = self.stn(x) # BxNxN
#         x = x.transpose(2, 1) # BxNxC
#         x = torch.bmm(x, trans)
#         x = x.transpose(2, 1) # BxCxN
#         x_1 = F.relu(self.conv1(x)) #Bx64xN
#         x_2 = F.relu(self.conv2(x_1)) #Bx128xN
#         x_3 = F.relu(self.conv2(x_2)) #Bx128xN
#         x_4 = F.relu(self.conv2(x_3)) #Bx128xN
#         x_5 = F.relu(self.conv2(x_4)) #Bx512xN
#
#         x_global = torch.max(x_5, 2, keepdim=True)[0] # BxCxN -> BxCx1
#         x = x_global.repeat(1, 1, n_pts) # Bx2048XN
#         x_feat = torch.cat([x_1, x_2, x_3, x_4, x_5, x, cls], 1) #Bx3024xN
#
#         return x_feat
class PointNetCls(nn.Module):
    def __init__(self, k=3, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1,2) # BxNxC -> BxCxN
        x_global, trans_feat = self.feat(x)
        x = F.relu(self.fc1(x_global))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x, x_global.unsqueeze(2), trans_feat

class PointNetSeg(nn.Module):
    def __init__(self, NUM_SEG_CLASSES):
        super(PointNetSeg, self).__init__()
        self.output_dim = NUM_SEG_CLASSES

        self.stn = STN3d()
        self.fstn = STNkd(k=128)

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 2048, 1)

        self.fc1 = torch.nn.Linear(3024, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, self.output_dim)

        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, cls):
        # x = x.transpose(2, 1)  # BxCxN, cls: Bx1xC'
        # cls = cls.transpose(2, 1) # BxC'x1
        x = x.transpose(1, 2)  # BxNxC -> BxCxN
        n_pts = x.size()[2]
        trans = self.stn(x)  # BxNxN
        x = x.transpose(2, 1)  # BxNxC
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)  # BxCxN
        x1 = F.relu(self.conv1(x)) # Bx64xN
        x2 = F.relu(self.conv2(x1)) # Bx128xN
        x3 = F.relu(self.conv3(x2)) # Bx128xN

        trans_feat = self.fstn(x3)
        x = torch.bmm(x3.transpose(2, 1), trans_feat)
        x = x.transpose(2, 1)

        x4 = F.relu(self.conv4(x)) #Bx128xN
        x5 = F.relu(self.conv5(x4)) #Bx512xN
        x6 = F.relu(self.conv6(x5)) #Bx2048xN

        x_global = torch.max(x6, 2, keepdim=True)[0] #Bx2048x1
        x_tile = x_global.repeat(1, 1, n_pts) # Bx2048xN
        cls_tile = cls.transpose(2, 1).repeat(1, 1, n_pts) # BxC'xN
        x_all = torch.cat((x1, x2, x3, x4, x5, x_tile, cls_tile), 1) #Bx3024xN

        x = x_all.transpose(1, 2) # BxNxC
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        x = x.transpose(1,2) # BxCxN

        return x, x_global, trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, num_classes=16, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.num_classes = num_classes
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.num_classes, 1)


    def forward(self, x):
        batchsize = x.size()[0] # BxNxC
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
