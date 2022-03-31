#!/usr/bin/env python3

"""
Created on Tue Jun 8 2021

@author: Oshada Jayasinghe
"""

import torch
import numpy as np
import torchvision

class backbone(torch.nn.Module):
    def __init__(self):
        super(backbone,self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class lane_net(torch.nn.Module):
    def __init__(self, size=(288, 800), cls_dim=(101, 18, 4)):
        super(lane_net, self).__init__()
        self.cls_dim = cls_dim # (no_of_gridding_cells, no_of_row_anchors, no_of_lanes)
        self.total_dim = np.prod(cls_dim)
        self.model = backbone()

        self.cls = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(2048, self.total_dim),
        )

        self.pool2 = torch.nn.Conv2d(256,8,1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = self.model(x)
        x = self.pool1(x)
        x = self.pool2(x).view(-1,1800)
        x = self.cls(x).view(-1, *self.cls_dim)

        return x