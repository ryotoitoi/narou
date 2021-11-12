import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet
from matplotlib import pyplot as plt

sys_paths = [
    "/home/yongtae/Lung_project/bonoo_jf_lung",
    "/Users/yongtae/Documents/bonbon/bonoo_jf_lung",
]

for sys_path in sys_paths:
    if sys_path not in sys.path:
        sys.path.append(sys_path)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SimpleAttentionResnetNetwork(nn.Module):
    def __init__(self, num_classes, base_model_path=None, pre_trained: bool = True):
        super().__init__()
        depth = 50
        n = 2
        block = Bottleneck if depth >= 44 else BasicBlock
        # base_model = models.resnet50(pretrained=True)

        base_model = None
        if base_model_path != None:
            base_model = torch.load(base_model_path, map_location="cpu")
        else:
            base_model = torchvision.models.resnet50(pretrained=pre_trained)
        self.features = nn.Sequential(*[layer for layer in base_model.children()][:-2])
        self.attn_resnet_block = self._make_layer(
            block, 2048, n, stride=1, down_size=False
        )
        self.attn_conv = nn.Sequential(nn.Conv2d(2048, 1, 1), nn.Sigmoid())
        self.fc = nn.Linear(2048, num_classes)
        self.mask_ = None

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        layers = []
        layers.append(block(2048, 1024, stride, downsample))
        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * 1
            for i in range(1, blocks):
                layers.append(block(inplanes, 1024))

            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)

        attn = self.attn_resnet_block(x)
        attn = self.attn_conv(attn)  # [B, 1, H, W]
        B, _, H, W = attn.shape
        self.mask_ = attn.detach().cpu()

        rx = x * attn
        rx = rx + x
        rx = F.adaptive_avg_pool2d(rx, (1, 1))
        rx = rx.reshape(B, -1)

        return self.fc(rx)

    def save_attention_mask(self, x, path):
        B = x.shape[0]
        self.forward(x)
        x = x.cpu() * torch.Tensor([0.7, 0.6, 0.7]).reshape(-1, 1, 1)
        x = x + torch.Tensor([0.15, 0.15, 0.15]).reshape(-1, 1, 1)
        fig, axs = plt.subplots(4, 2, figsize=(6, 8))
        plt.axis("off")
        for i in range(4):
            axs[i, 0].imshow(x[i].permute(1, 2, 0))
            axs[i, 1].imshow(self.mask_[i][0])
        plt.savefig(path)
        plt.close()

    def output_attention_mask(self, x):
        B = x.shape[0]
        prediction = self.forward(x)
        return self.mask_, prediction


class SimpleAttentionEfficientNetwork(nn.Module):
    def __init__(self, num_classes, base_model_path=None, pre_trained: bool = True):
        super().__init__()
        depth = 50
        n = 2
        block = Bottleneck if depth >= 44 else BasicBlock
        # base_model = models.resnet50(pretrained=True)

        self.base_model = None
        if base_model_path != None:
            self.base_model = torch.load(base_model_path, map_location="cpu")
        else:
            if pre_trained:
                self.base_model = EfficientNet.from_pretrained("efficientnet-b4")
            else:
                self.base_model = EfficientNet.from_name("efficientnet-b4")
        self.attn_resnet_block = self._make_layer(
            block, 2048, n, stride=1, down_size=False
        )
        self.attn_conv = nn.Sequential(nn.Conv2d(2048, 1, 1), nn.Sigmoid())
        self.fc = nn.Linear(2048, num_classes)
        self.mask_ = None

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        layers = []
        layers.append(block(2048, 1024, stride, downsample))
        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * 1
            for i in range(1, blocks):
                layers.append(block(inplanes, 1024))

            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base_model.extract_features(x)

        attn = self.attn_resnet_block(x)
        attn = self.attn_conv(attn)  # [B, 1, H, W]
        B, _, H, W = attn.shape
        self.mask_ = attn.detach().cpu()

        rx = x * attn
        rx = rx + x
        rx = F.adaptive_avg_pool2d(rx, (1, 1))
        rx = rx.reshape(B, -1)

        return self.fc(rx)

    def save_attention_mask(self, x, path):
        B = x.shape[0]
        self.forward(x)
        x = x.cpu() * torch.Tensor([0.7, 0.6, 0.7]).reshape(-1, 1, 1)
        x = x + torch.Tensor([0.15, 0.15, 0.15]).reshape(-1, 1, 1)
        fig, axs = plt.subplots(4, 2, figsize=(6, 8))
        plt.axis("off")
        for i in range(4):
            axs[i, 0].imshow(x[i].permute(1, 2, 0))
            axs[i, 1].imshow(self.mask_[i][0])
        plt.savefig(path)
        plt.close()

    def output_attention_mask(self, x):
        B = x.shape[0]
        prediction = self.forward(x)
        return self.mask_, prediction
