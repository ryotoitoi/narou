#%%

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet
from matplotlib import pyplot as plt


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
        self.fc = nn.Linear(2048 + 768, num_classes)
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


#%%
class SimpleAttentionEfficientNetwork(nn.Module):
    def __init__(
        self,
        num_classes: int = 768,
        base_model_path: Optional[str] = None,
        pre_trained: bool = True,
        transfer_learning: bool = False,
        model_name_num: int = 1,
        image_size: int = 224,
    ):
        """
        Efficient-NetにAttention mechanismを追加したモデル

        Args:
            num_classes (int, optional): 予想するクラスの数. Defaults to 768.
            base_model_path ([type], optional): loadしたいモデルのpath. Defaults to None.
            pre_trained (bool, optional): 事前学習済みモデルを利用するのか. Defaults to True.
            transfer_learning (bool, optional): 転移学習をするのか. Defaults to False.
            model_name_num (int, optional): effientnet-b{ここの数}. Defaults to 1.
            image_size (int, optional): input_imageのサイズ. Defaults to 224.
        """
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
                self.base_model = EfficientNet.from_pretrained(
                    "efficientnet-b{}".format(model_name_num)
                )
            else:
                self.base_model = EfficientNet.from_name(
                    "efficientnet-b{}".format(model_name_num)
                )
        if transfer_learning:
            for param in self.base_model.parameters():
                param.requires_grad = False
        dummy_data = torch.ones(
            (1, 3, image_size, image_size)
        )  # TODO ここのサイズはsettingみたいなので、行えるようにしたい
        data = self.base_model.extract_features(dummy_data)
        self.feature_number = data.shape[1]
        self.attn_resnet_block = self._make_layer(
            block, self.feature_number, n, stride=1, down_size=False
        )
        self.attn_conv = nn.Sequential(
            nn.Conv2d(self.feature_number, 1, 1), nn.Sigmoid()
        )
        self.fc = nn.Linear(self.feature_number, num_classes)
        self.mask_ = None

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        layers = []
        layers.append(block(self.feature_number, 1024, stride, downsample))
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

    def get_features(self, x):
        x = self.base_model.extract_features(x)

        attn = self.attn_resnet_block(x)
        attn = self.attn_conv(attn)  # [B, 1, H, W]
        B, _, H, W = attn.shape
        self.mask_ = attn.detach().cpu()
        rx = x * attn
        rx = rx + x
        rx = F.adaptive_avg_pool2d(rx, (1, 1))
        return rx.reshape(B, -1)  # * [1,1280]のベクトル

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

    def output_attention_mask_with_prediction(self, x):
        B = x.shape[0]
        prediction = self.forward(x)
        return self.mask_, prediction

    def output_attention_mask_with_features(self, x):
        B = x.shape[0]
        features = self.get_features(x)
        return self.mask_, features


# %%
