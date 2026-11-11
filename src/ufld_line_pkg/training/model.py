"""Compact UFLD-v2-style lane model.

Outputs exactly the four tensors the C++ decoder (ufld_decoder.cpp) reads, with
the names ufld_line.launch expects:

    loc_row   : [B, num_grid_row, num_row, num_lanes]
    loc_col   : [B, num_grid_col, num_col, num_lanes]
    exist_row : [B, 2,            num_row, num_lanes]
    exist_col : [B, 2,            num_col, num_lanes]

A ResNet-18 backbone (ImageNet-pretrained if cached) feeds a 1x1 channel
reducer + flatten + linear heads, mirroring UFLD-v2's head structure but small
enough to train and to convert to TNN for the D3-G board.
"""
import torch
import torch.nn as nn
import torchvision


class UfldNet(nn.Module):
    def __init__(self, num_lanes=4, num_row=56, num_col=41,
                 num_grid_row=100, num_grid_col=100, pretrained=True, head_dim=256):
        super().__init__()
        self.NL, self.NR, self.NC = num_lanes, num_row, num_col
        self.GR, self.GC = num_grid_row, num_grid_col

        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            backbone = torchvision.models.resnet18(weights=weights)
        except Exception:
            backbone = torchvision.models.resnet18(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        # For 160x320 input, layer4 output is 512 x 5 x 10.
        self.reduce = nn.Conv2d(512, 8, 1)
        feat = 8 * 5 * 10
        self.shared = nn.Sequential(nn.Linear(feat, head_dim), nn.ReLU(inplace=True))
        self.fc_loc_row = nn.Linear(head_dim, self.GR * self.NR * self.NL)
        self.fc_loc_col = nn.Linear(head_dim, self.GC * self.NC * self.NL)
        self.fc_exist_row = nn.Linear(head_dim, 2 * self.NR * self.NL)
        self.fc_exist_col = nn.Linear(head_dim, 2 * self.NC * self.NL)

    def forward(self, x):
        f = self.stem(x)
        f = self.reduce(f).flatten(1)
        h = self.shared(f)
        B = x.shape[0]
        loc_row = self.fc_loc_row(h).view(B, self.GR, self.NR, self.NL)
        loc_col = self.fc_loc_col(h).view(B, self.GC, self.NC, self.NL)
        exist_row = self.fc_exist_row(h).view(B, 2, self.NR, self.NL)
        exist_col = self.fc_exist_col(h).view(B, 2, self.NC, self.NL)
        return loc_row, loc_col, exist_row, exist_col


# ImageNet normalisation — must match tnn_ufld_runner.cpp preprocessing.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
