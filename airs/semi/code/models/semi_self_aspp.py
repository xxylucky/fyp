import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from utils.aug_function import FeatureNoiseDecoder, DropOutDecoder


def cat(x1, x2, x3=None, dim=1):
    if x3 is None:
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim)
        return x
    else:
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim)
        diffY = x.size()[2] - x3.size()[2]
        diffX = x.size()[3] - x3.size()[3]
        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x3], dim=1)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MultiScaleConvBlock(nn.Module):
    """
    轻量多尺度卷积模块：
    branch1: 1x1
    branch2: 3x3
    branch3: 5x5
    branch4: 3x3 dilation=2
    最后 concat + 1x1 fuse，并加残差
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvBlock, self).__init__()

        mid_channels = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.fuse(out)

        res = self.residual(x)
        out = out + res
        return F.relu(out, inplace=True)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        if transpose:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.dropout = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        resnet = models.resnet34(pretrained=True)

        if in_channels == 3:
            self.encoder1_conv = resnet.conv1
        else:
            self.encoder1_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1   # 64
        self.encoder3 = resnet.layer2   # 128
        self.encoder4 = resnet.layer3   # 256
        self.encoder5 = resnet.layer4   # 512

    def forward(self, x):
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_maxpool = self.maxpool(e1)

        e2 = self.encoder2(e1_maxpool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        return e1, e2, e3, e4, e5


class MyModel(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels)

        # ===== 新增：bottleneck 多尺度卷积模块 =====
        self.ms_bottleneck = MultiScaleConvBlock(512, 512)

        # seg-Decoder
        self.segDecoder5 = DecoderBlock(512, 512)
        self.segDecoder4 = DecoderBlock(512 + 256, 256)
        self.segDecoder3 = DecoderBlock(256 + 128, 128)
        self.segDecoder2 = DecoderBlock(128 + 64, 64)
        self.segDecoder1 = DecoderBlock(64 + 64, 64)

        self.segconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, 1)
        )

        # inp-Decoder
        self.inpDecoder5 = DecoderBlock(512, 512, transpose=True)
        self.inpDecoder4 = DecoderBlock(512 + 256, 256, transpose=True)
        self.inpDecoder3 = DecoderBlock(256 + 128, 128, transpose=True)
        self.inpDecoder2 = DecoderBlock(128 + 64, 64, transpose=True)
        self.inpDecoder1 = DecoderBlock(64 + 64, 64, transpose=True)

        self.inpSideout5 = SideoutBlock(512, 1)
        self.inpSideout4 = SideoutBlock(256, 1)
        self.inpSideout3 = SideoutBlock(128, 1)
        self.inpSideout2 = SideoutBlock(64, 1)

        self.inpconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, 1)
        )

        self.dropout = DropOutDecoder()

    def forward(self, x):
        # ========= Encoder =========
        e1, e2, e3, e4, e5 = self.encoder(x)

        # ========= 新增：对最深层特征做多尺度增强 =========
        e5_ms = self.ms_bottleneck(e5)

        # ========= Seg branch =========
        d5 = self.segDecoder5(e5_ms)
        d4 = self.segDecoder4(cat(d5, e4))
        d3 = self.segDecoder3(cat(d4, e3))
        d2 = self.segDecoder2(cat(d3, e2))
        d1 = self.segDecoder1(cat(d2, e1))

        mask = self.segconv(d1)
        mask = torch.sigmoid(mask)
        mask_binary = (mask > 0.5).float()

        # ========= Inpaint / prior-guided branch =========
        inpe1, inpe2, inpe3, inpe4 = e1, e2, e3, e4
        inpe5 = self.dropout(e5_ms)

        inpd5 = self.inpDecoder5(inpe5)
        inpimg5 = torch.sigmoid(self.inpSideout5(inpd5))

        inpd4 = self.inpDecoder4(cat(inpd5, inpe4))
        inpimg4 = torch.sigmoid(self.inpSideout4(inpd4))

        inpd3 = self.inpDecoder3(cat(inpd4, inpe3))
        inpimg3 = torch.sigmoid(self.inpSideout3(inpd3))

        inpd2 = self.inpDecoder2(cat(inpd3, inpe2))
        inpimg2 = torch.sigmoid(self.inpSideout2(inpd2))

        inpd1 = self.inpDecoder1(cat(inpd2, inpe1))
        preboud = torch.sigmoid(self.inpconv(inpd1))

        return mask, preboud, inpimg2, inpimg3, inpimg4, inpimg5, mask_binary