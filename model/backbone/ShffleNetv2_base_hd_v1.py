"""shufflenetv2 in pytorch



[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun

    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.Common_fun import *


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels // groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = int(in_channels / 2)
        # 1*1 Conv
        self.primary_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True))

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )

            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, int(out_channels / 2), 1),
            )

        else:
            
            # main branch
            self.residual = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.shortcut = nn.Sequential()

    def forward(self, x):
        
        primary_conv = self.primary_conv(x)
        shortcut = self.shortcut(primary_conv)
        residual = self.residual(primary_conv)
        x1 = torch.cat([shortcut, residual], dim=1)

        if self.stride == 1 and self.out_channels == self.in_channels:
            x1 = x + x1  # residual

        x = channel_shuffle(x1, 2)

        return x


class ShuffleNetV2(nn.Module):

    def __init__(self, class_num=100, input_c = 3):
        super().__init__()
        stage_layers = [2, 6, 3]

        out_channels = [16, 32, 48, 64]

        init_c = 24

        self.pre = nn.Sequential(
            nn.Conv2d(input_c, init_c, 3, stride=2,padding=1),
            nn.BatchNorm2d(init_c)
        )

        self.stage2 = self._make_stage(init_c, out_channels[0], stage_layers[0])
        self.se2 = SELayer(out_channels[0])
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], stage_layers[1])
        self.se3 = SELayer(out_channels[1])
        self.stage4 = self._make_stage(out_channels[1], out_channels[2],stage_layers[2])
        self.se4 = SELayer(out_channels[2])
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.stage2(x)
        x = self.se2(x)
        x = self.stage3(x)
        x = self.se3(x)
        x = self.stage4(x)
        x = self.se4(x)
        x = self.conv5(x)

        return x

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1

        return nn.Sequential(*layers)

def shufflenetv2():
    return ShuffleNetV2(input_c = 3)