"""CIFAR ResNet-18 with noise-injectable forward for ANSD.

forward(x, noise=False, noise_lambda=0.0) returns (features, logits) where
features = [f1, f2, f3, f4] are the 4 residual-block outputs. When noise=True,
adaptive Gaussian noise (Eq7-8) is injected into the Block-1 feature f1 before
it propagates to deeper blocks — this builds the noisy STUDENT view.
[SUMBER: ANSD raw md §3.2 Eq7-8, Table 2 (Block1 = best injection point)]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.adaptive_noise import adaptive_noise


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


class CIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 64
        # CIFAR stem: 3x3 conv, no maxpool (32x32 input)
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            downsample = None
            if s != 1 or self.in_planes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.in_planes, planes * block.expansion, s),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers.append(block(self.in_planes, planes, s, downsample))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, noise=False, noise_lambda=0.0):
        x = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(x)
        if noise:
            f1 = adaptive_noise(f1, noise_lambda)  # inject at Block-1 (Eq8)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        out = self.avgpool(f4).flatten(1)
        logits = self.fc(out)
        return [f1, f2, f3, f4], logits


def CIFAR_ResNet18(num_classes=100):
    return CIFAR_ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def get_network(args):
    if args.data_type in ('cifar100', 'cifar10'):
        num_classes = 100 if args.data_type == 'cifar100' else 10
        if args.classifier_type == 'ResNet18':
            return CIFAR_ResNet18(num_classes=num_classes)
        raise NotImplementedError(args.classifier_type)
    raise NotImplementedError(args.data_type)
