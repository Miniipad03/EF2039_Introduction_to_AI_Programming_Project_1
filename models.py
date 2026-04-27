import torch
import torch.nn as nn
from torchvision import models


# ── Attention Modules ─────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[:2]
        avg = x.mean(dim=[2, 3])
        mx  = x.amax(dim=[2, 3])
        return x * self.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))


class BasicBlockWithAttention(nn.Module):
    """BasicBlock 내부 residual path에만 attention 적용 (skip 더하기 전)."""
    def __init__(self, block, attn):
        super().__init__()
        self.conv1      = block.conv1
        self.bn1        = block.bn1
        self.relu       = block.relu
        self.conv2      = block.conv2
        self.bn2        = block.bn2
        self.downsample = block.downsample
        self.stride     = block.stride
        self.attn       = attn

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attn(out)            # skip 더하기 전에 적용
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def inject_attention(layer, attn_type):
    if attn_type == 'none':
        return
    for i, block in enumerate(layer):
        channels = block.conv2.out_channels
        if attn_type == 'channel':
            attn = ChannelAttention(channels)
        elif attn_type == 'spatial':
            attn = SpatialAttention()
        else:  # cbam
            attn = CBAM(channels)
        layer[i] = BasicBlockWithAttention(block, attn)


# ── Models ────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model_dropout(num_classes, attn='none', dropout=0.5):
    """ResNet-34 with Dropout inserted before the FC layer (avgpool → dropout → fc).
    Pretrained backbone은 그대로 유지하고 head만 교체.
    """
    m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    in_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    if attn != 'none':
        for name in ('layer3', 'layer4'):
            inject_attention(getattr(m, name), attn)
        n = sum(1 for mod in m.modules() if isinstance(mod, BasicBlockWithAttention))
        print(f"  resnet34_dropout(p={dropout}) + {attn.upper()}  injected={n} modules")
    else:
        print(f"  resnet34_dropout(p={dropout}, no attention)")
    return m


def build_model(model_name, num_classes, attn='none'):
    if model_name == 'cnn':
        m = SimpleCNN(num_classes)
        print(f"  SimpleCNN  params={sum(p.numel() for p in m.parameters()):,}")
        return m

    weights = (models.ResNet18_Weights.DEFAULT if model_name == 'resnet18'
               else models.ResNet34_Weights.DEFAULT)
    m = (models.resnet18(weights=weights) if model_name == 'resnet18'
         else models.resnet34(weights=weights))
    m.fc = nn.Linear(m.fc.in_features, num_classes)

    if attn != 'none':
        for name in ('layer3', 'layer4'):
            inject_attention(getattr(m, name), attn)
        n = sum(1 for mod in m.modules() if isinstance(mod, BasicBlockWithAttention))
        print(f"  {model_name} + {attn.upper()}  injected={n} modules")
    else:
        print(f"  {model_name} (no attention)")
    return m
