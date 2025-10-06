# -*- coding: utf-8 -*-
"""
Adapted from: https://github.com/wyharveychen/CloserLookFewShot
This file contains Conv32F(ReLU/LeakyReLU), Conv64F(ReLU/LeakyReLU) and R2D2Embedding.
"""

import torch
import torch.nn as nn
import numpy as np

def floor_power(num, divisor, power):
    """Performs what we call a floor power, a recursive fixed division process
        with a flooring between each time

    Args:
        num (int or float):The original number to divide from
        divisor (int or float): The actual divisor for the number
        power (int): How many times we apply this divide and then floor

    Returns:
        int: The numerical result of the floor division process
    """
    for _ in range(power):
        num = np.floor(num/divisor)
    return num


class Conv64F(nn.Module):
    """
    Four convolutional blocks network, each of which consists of a Covolutional layer,
    a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
    Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

    Input:  3 * 84 *84
    Output: 64 * 5 * 5
    """

    def __init__(
        self,
        is_flatten=False,
        is_feature=False,
        leaky_relu=False,
        negative_slope=0.2,
        last_pool=True,
        maxpool_last2=True,
        use_running_statistics=True,
        num_channels=3,
    ):
        super(Conv64F, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature
        self.last_pool = last_pool
        self.maxpool_last2 = maxpool_last2

        if leaky_relu:
            activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        else:
            activation = nn.ReLU(inplace=False)

        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, track_running_stats=use_running_statistics),
            activation,
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, track_running_stats=use_running_statistics),
            activation,
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, track_running_stats=use_running_statistics),
            activation,
        )
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=3, stride=3)

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, track_running_stats=use_running_statistics),
            activation,
        )
        self.layer4_pool = nn.MaxPool2d(kernel_size=3, stride=3)
        
        num_logits = int(64 * floor_power(128, 3, 4) * floor_power(157, 3, 4))
        self.logits = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(num_logits, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(in_features=num_logits, out_features=1600)
        )

        # Xavier initialization for the Linear layer
        # nn.init.xavier_normal_(self.logits[1].weight)
        # if self.logits[1].bias is not None:
        #     nn.init.zeros_(self.logits[1].bias)

    def forward(self, x):
        # print('Input shape:', x.shape)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        
        # print('out1 shape:', out1.shape)
        # print('out2 shape:', out2.shape)
        # print('out3 shape:', out3.shape)

        if self.maxpool_last2:
            out3 = self.layer3_maxpool(out3)  # for some methods(relation net etc.)
            # print('out3 shape after maxpool:', out3.shape)

        out4 = self.layer4(out3)
        if self.last_pool:
            out4 = self.layer4_pool(out4)

        if self.is_flatten:
            out4 = out4.view(out4.size(0), -1)
            out4 = self.logits(out4)
            

        # print('out4 shape:', out4.shape)
        # exit()

        if self.is_feature:
            return out1, out2, out3, out4

        return out4


class Conv32F(nn.Module):
    """
    Four convolutional blocks network, each of which consists of a Covolutional layer,
    a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
    Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

    Input:  3 * 84 *84
    Output: 32 * 5 * 5
    """

    def __init__(
        self,
        is_flatten=False,
        is_feature=False,
        leaky_relu=False,
        negative_slope=0.2,
        last_pool=True,
        num_channels=3
    ):
        super(Conv32F, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature
        self.last_pool = last_pool

        if leaky_relu:
            activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        else:
            activation = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            activation,
        )
        self.layer4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        if self.last_pool:
            out4 = self.layer4_pool(out4)

        if self.is_flatten:
            out4 = out4.view(out4.size(0), -1)

        if self.is_feature:
            return out1, out2, out3, out4

        return out4


def R2D2_conv_block(
    in_channels,
    out_channels,
    retain_activation=True,
    keep_prob=1.0,
    pool_stride=2,
):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2, stride=pool_stride),
    )
    if retain_activation:
        block.add_module("LeakyReLU", nn.LeakyReLU(0.1))

    if keep_prob < 1.0:
        block.add_module("Dropout", nn.Dropout(p=1 - keep_prob, inplace=False))

    return block


class R2D2Embedding(nn.Module):
    """
    https://github.com/kjunelee/MetaOptNet/blob/master/models/R2D2_embedding.py
    """

    def __init__(
        self,
        x_dim=3,
        h1_dim=96,
        h2_dim=192,
        h3_dim=384,
        z_dim=512,
        retain_last_activation=False,
    ):
        super(R2D2Embedding, self).__init__()

        self.block1 = R2D2_conv_block(x_dim, h1_dim)
        self.block2 = R2D2_conv_block(h1_dim, h2_dim)
        self.block3 = R2D2_conv_block(h2_dim, h3_dim, keep_prob=0.9)
        # In the last conv block, we disable activation function to boost the classification accuracy.
        # This trick was proposed by Gidaris et al. (CVPR 2018).
        # With this trick, the accuracy goes up from 50% to 51%.
        # Although the authors of R2D2 did not mention this trick in the paper,
        # we were unable to reproduce the result of Bertinetto et al. without resorting to this trick.
        self.block4 = R2D2_conv_block(
            h3_dim,
            z_dim,
            retain_activation=retain_last_activation,
            keep_prob=0.9,
            pool_stride=1,
        )

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        # Flatten and concatenate the output of the 3rd and 4th conv blocks as proposed in R2D2 paper.
        return torch.cat((b3.view(b3.size(0), -1), b4.view(b4.size(0), -1)), 1)
