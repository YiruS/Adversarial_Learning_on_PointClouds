from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class ConvDiscNet(nn.Module):
    def __init__(self, input_dim):
        super(ConvDiscNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 16, 1)
        self.fc = nn.Linear(16, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.transpose(2, 1) # (BxNxC) -> (BxCxN)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(2, 1)  # (BxNxC')
        x = self.fc(x) # (BxNx1)
        x = x.squeeze(2)
        return x

class DeepConvDiscNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepConvDiscNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 64, 1)
        self.conv5 = torch.nn.Conv1d(64, 64, 1)
        self.fc = nn.Linear(64, output_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x = x.unsqueeze(2)  # BxCx1
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        x = x.view(-1, 64)
        x = self.fc(x) # (BxNx1)
        return x

class PointwiseDiscNet(nn.Module):
    def __init__(self, input_pts, input_dim):
        super(PointwiseDiscNet, self).__init__()
        self.input_pts = input_pts

        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1) # 64
        self.conv2 = torch.nn.Conv1d(64, 64, 1) # 64
        self.conv3 = torch.nn.Conv1d(64, 64, 1) # 128
        self.conv4 = torch.nn.Conv1d(64, 128, 1)  # 128

    def forward(self, x):
        # x = x.transpose(2, 1) # BxCxN
        x = F.relu(self.conv1(x)) # Bx50xN
        x = F.relu(self.conv2(x)) # Bx64xN
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.transpose(2,1) # BxNxC
        x = torch.max(x, 2, keepdim=True)[0]  # BxNx1
        x = x.view(-1, self.input_pts)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.dropout(self.fc3(x))
        # x = self.dropout(self.fc4(x))
        # x = self.fc5(x)

        return x


class BaseDiscNet(nn.Module):
    def __init__(self, input_pts, input_dim, output_dim):
        super(BaseDiscNet, self).__init__()
        self.input_pts = input_pts

        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)  # 64
        self.conv2 = torch.nn.Conv1d(64, 64, 1)  # 64
        self.conv3 = torch.nn.Conv1d(64, output_dim, 1)  # 128
        self.conv4 = torch.nn.Conv1d(output_dim, output_dim, 1)  # 128

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))  # Bx50xN
        x = self.leaky_relu(self.conv2(x))  # Bx64xN
        x = self.leaky_relu(self.conv3(x))
        return x

class ShapeDiscNet(nn.Module):
    def __init__(self, shared_output_dim, num_shapes):
        super(ShapeDiscNet, self).__init__()
        self.interm_dim = 512

        self.conv = torch.nn.Conv1d(shared_output_dim, self.interm_dim, 1)
        self.fc1 = torch.nn.Linear(self.interm_dim, 64)
        self.fc2 = torch.nn.Linear(64, num_shapes)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv(x))
        x = torch.max(x, 2, keepdim=False)[0]  # BxCx1
        x = x.view(-1, self.interm_dim)
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PointDiscNet(nn.Module):
    def __init__(self, shared_output_dim, input_pts):
        super(PointDiscNet, self).__init__()
        self.input_pts = input_pts

        self.conv1 = torch.nn.Conv1d(shared_output_dim, 256, 1)  # 128
        self.conv2 = torch.nn.Conv1d(256, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = x.transpose(2, 1)  # BxNxC
        x = torch.max(x, 2, keepdim=True)[0]  # BxNx1
        x = x.view(-1, self.input_pts)
        return x

class StackDiscNet(nn.Module):
    def __init__(self, input_pts, input_dim, num_shapes):
        super(StackDiscNet, self).__init__()
        self.input_pts = input_pts

        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1) # 64
        self.conv2 = torch.nn.Conv1d(64, 64, 1) # 64
        self.conv3 = torch.nn.Conv1d(64, 64, 1) # 128
        self.conv4 = torch.nn.Conv1d(64, 128, 1)  # 128

        self.conv5 = torch.nn.Conv1d(1, num_shapes, 1)
        # self.disc = torch.nn.Linear(num_shapes, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def custom_activation(self, x):
        # logexpsum = backend.sum(backend.exp(x), axis=-1, keepdims=True)
        x = x.transpose(2, 1) # BxNxS
        out = torch.logsumexp(x, dim=2, keepdim=True)
        res = out / (out + 1.0)
        return res


    def forward(self, x):
        x = self.leaky_relu(self.conv1(x)) # Bx50xN
        x = self.leaky_relu(self.conv2(x)) # Bx64xN
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x)) # BxCxN

        # x = x.transpose(2,1) # BxNxC
        x = torch.max(x, 1, keepdim=True)[0]  # Bx1xN
        # x = x.view(-1, self.input_pts)
        shape_logits = self.conv5(x) # BxSxN

        disc_out = self.custom_activation(shape_logits) # BxNx1

        return shape_logits, disc_out