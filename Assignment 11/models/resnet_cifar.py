import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock as ResBlock

class LayerBlock(nn.Module):
    def __init__(self, in_planes, out_planes, inc_res_block, inc_pool):
        super(LayerBlock, self).__init__()
        self.inc_res_block = inc_res_block
        self.conv = self._make_conv(in_planes, out_planes, inc_pool)
        self.res = self._make_layer(out_planes, out_planes)
    
    def _make_conv(self, in_planes, out_planes, inc_pool):
        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                    padding=1, bias=False)
        ]
        if inc_pool:
            layers.append(nn.MaxPool2d(2, 2))
        layers.extend([
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
        ])
        return nn.Sequential(*layers)

    def _make_layer(self, in_planes, out_planes):
        layers = []
        layers.append(ResBlock(in_planes, out_planes, stride=1, dropout=0.0))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.inc_res_block:
            rout = self.res(out)
            out = rout + out
        return out

class ResNetCifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCifar10, self).__init__()
        self.layer0 = self._make_layer(3, 64, False, False)
        self.layer1 = self._make_layer(64, 128, True, True)
        self.layer2 = self._make_layer(128, 256, False, True)
        self.layer3 = self._make_layer(256, 512, True, True)
        self.pool = nn.MaxPool2d(4)
        self.linear = nn.Linear(512, num_classes, bias=False)

    def _make_layer(self, in_planes, out_planes, inc_res_block, inc_pool):
        layers = []
        layers.append(LayerBlock(in_planes, out_planes, inc_res_block, inc_pool))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = out.view(-1, 10)
        return F.log_softmax(out, dim=-1)
