from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


from src.nets.pointnet_utils import PointNetSetAbstraction
"""
Source: https://github.com/fxia22/pointnet.pytorch/blob/f0c2430b0b1529e3f76fb5d6cd6ca14be763d975/pointnet/model.py
"""


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

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        )
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
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, input_dim, shallow_dim, mid_dim, out_dim, global_feat=False):
        super(PointNetfeat, self).__init__()
        self.shallow_layer = nn.Sequential(
            nn.Conv1d(input_dim, shallow_dim, 1), nn.BatchNorm1d(shallow_dim)
        )

        self.base_layer = nn.Sequential(
            nn.Conv1d(shallow_dim, mid_dim, 1),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Conv1d(mid_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
        )

        self.global_feat = global_feat
        self.out_dim = out_dim

    def forward(self, x):
        n_pts = x.size()[2]
        x = self.shallow_layer(x)
        pointfeat = x

        x = self.base_layer(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_dim)

        trans_feat = None
        trans = None
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.out_dim, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNet2(nn.Module):
    def __init__(self, in_channel, normal_channel=True):
        super(PointNet2, self).__init__()
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=778, radius=0.2, nsample=16, in_channel=in_channel, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=388, radius=0.4, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=194, radius=0.8, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 256, 256], group_all=True)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 256, 512], group_all=True)
    def forward(self, xyz):
        B, _, _ = xyz.shape#16,6,778
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)#16,3,512,16,128,512
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)#16,3,128,16,256,128
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)#16,1024,1
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)#16,1024,1
        # x = l4_points.view(B, 256)
        x = l4_points.view(B, 512)
        return x
    
class ReconDecoder(nn.Module):
    def __init__(self, input_dim = 256+3):
        super(ReconDecoder, self).__init__()
        self.sa1 = PointNetfeat(input_dim=input_dim, shallow_dim = 64, mid_dim = 64, out_dim = 64)
        self.sa2 = PointNetfeat(input_dim=128, shallow_dim = 32, mid_dim = 32, out_dim = 32)
        self.sa3 = PointNetfeat(input_dim=64, shallow_dim = 16, mid_dim = 16, out_dim = 16)
        self.sa4 = PointNetfeat(input_dim=32, shallow_dim = 8, mid_dim = 8, out_dim = 8)
        self.head = nn.Linear(16,3)

    def forward(self, x):
        x = self.sa1(x)[0]
        x = self.sa2(x)[0]
        x = self.sa3(x)[0]
        x = self.sa4(x)[0].permute(0,2,1)
        x = self.head(x)
        return x