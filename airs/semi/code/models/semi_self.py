import torch
import torch.nn as nn  # 带参数的“层”
import torchvision.models as models
import torch.nn.functional as F  # 对比nn：不带参数的函数


from utils.aug_function import FeatureNoiseDecoder, DropOutDecoder


def cat(x1, x2, x3=None, dim=1):
    # Concatenate，级联/拼接。
    # 自动修建/补齐，把不同尺寸的矩阵叠在一起形成更高维的矩阵。
    if x3 == None:
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim)
        return x
    else:
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim)
        diffY = torch.tensor([x.size()[2] - x3.size()[2]])
        diffX = torch.tensor([x.size()[3] - x3.size()[3]])
        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x3], dim=1)
        return x


class ConvBlock(nn.Module):
    # 相当于一个连续执行 filter2 -> norm -> max(0, x) 的自定义函数。
    # 卷积 (Conv) → 归一化 (Batch Norm) → 激活 (ReLU)
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    # 特征精炼 (Two ConvBlocks) + 尺寸放大 (Upsample)
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        # 尺寸放大：两种方法
        if transpose:  # 转置卷积
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(out_channels,
                                #    in_channels // 4,
                                   out_channels,
                                #    in_channels // 4,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias=False),
                # nn.BatchNorm2d(in_channels // 4),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:  # 双线性插值
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        # 数据流
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)

        return x


class SideoutBlock(nn.Module):
    # 从中间层的特征矩阵中，“偷窥”一下当前的预测结果长什么样。
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        # 卷积精炼：减少计算量，同时提炼核心特征
        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        # 随机让一部分神经元关闭（设为0）：防止过拟合
        self.dropout = nn.Dropout2d(0.1)

        # 1x1卷积，映射输出
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class Encoder(nn.Module):
    # 编码器：ResNet34
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        """
        编码器（Backbone）：用 ResNet34 抽取多尺度特征。

        作用（给新手看的理解版本）：
        - 输入一张图片/特征图 `x`（形状通常是 B x C x H x W）
        - 经过 ResNet34 的不同 stage，得到从浅到深的 5 级特征：e1~e5
        - 这些多尺度特征会被后面的 Decoder 用来做上采样 + 跳跃连接（skip connection）

        参数：
        - in_channels: 输入通道数。RGB 图像一般是 3；如果不是 3，这里会自己新建一个 conv1。
        """
        resnet = models.resnet34(pretrained=True)  # 调用resnet34框架并从网络上下载最新的权重
        # 从本地加载预训练权重
        # resnet.load_state_dict(torch.load("pretrain/backbone/resnet34-333f7ec4.pth"))

        if in_channels == 3:
            # 输入是 RGB：直接复用 ResNet34 的第一层卷积（conv1）
            self.encoder1_conv = resnet.conv1
        else:
            # 输入通道是灰度图或其他：自己定义一个新的 conv1（这一层没有预训练权重，会随机初始化）
            self.encoder1_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool  # 池化层
        # ResNet 的 4 个 stage（layer1~layer4），越往后语义越强、空间分辨率越小
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

    def forward(self, x):
        """
        输入：
        - x: (B, C, H, W)

        输出：
        - e1: (B, 64,  H/2,  W/2)   经过 conv1+bn+relu（浅层边缘/纹理）
        - e2: (B, 64,  H/4,  W/4)   layer1
        - e3: (B, 128, H/8,  W/8)   layer2
        - e4: (B, 256, H/16, W/16)  layer3
        - e5: (B, 512, H/32, W/32)  layer4（深层语义）
        """
        # 第 1 层：7x7 卷积 + stride=2，空间尺寸减半
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)

        # 池化层：进一步压缩尺寸再次下采样（通常从 H/2 -> H/4）
        e1_maxpool = self.maxpool(e1)

        # ResNet 四个 stage：逐级提取更抽象的特征
        e2 = self.encoder2(e1_maxpool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # 返回所有中间结果，为了和Decoder进行拼接
        return e1, e2, e3, e4, e5


class MyModel(nn.Module):
    def __init__(self,
                 num_classes=1,
                 in_channels=3
                 ):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels)
        
        # seg-Decoder
        self.segDecoder5 = DecoderBlock(512, 512)
        self.segDecoder4 = DecoderBlock(512 + 256, 256)
        self.segDecoder3 = DecoderBlock(256 + 128, 128)
        self.segDecoder2 = DecoderBlock(128 + 64, 64)
        self.segDecoder1 = DecoderBlock(64 + 64, 64)

        self.segconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(32, num_classes, 1))

        # inpaint-Decoder
        # 取消转置卷积
        self.inpDecoder5 = DecoderBlock(512, 512, transpose=False)
        self.inpDecoder4 = DecoderBlock(512 + 256, 256, transpose=False)
        self.inpDecoder3 = DecoderBlock(256 + 128, 128, transpose=False)
        self.inpDecoder2 = DecoderBlock(128 + 64, 64, transpose=False)
        self.inpDecoder1 = DecoderBlock(64 + 64, 64, transpose=False)

        self.inpSideout5 = SideoutBlock(512, 1)
        self.inpSideout4 = SideoutBlock(256, 1)
        self.inpSideout3 = SideoutBlock(128, 1)
        self.inpSideout2 = SideoutBlock(64, 1)

        self.inpconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(32, num_classes, 1))

        self.dropout = DropOutDecoder()      

    def forward(self, x):

        """这两行没用到"""
        # ori = x  # 原始输入图片
        # bs, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]  # 拿到输入图片的 批大小(bs), 通道数(C), 高度(H), 宽度(W)
        """Seg-branch"""
        e1, e2, e3, e4, e5 = self.encoder(x)  # ResNet34提取的5层特征

        # 逐步解码并放大（从 8x8 放大回 256x256）
        d5 = self.segDecoder5(e5)
        d4 = self.segDecoder4(cat(d5, e4))
        d3 = self.segDecoder3(cat(d4, e3))
        d2 = self.segDecoder2(cat(d3, e2))
        d1 = self.segDecoder1(cat(d2, e1))  # 此时 d1 已经回到了原始分辨率大小
        
        # 生成最终Mask
        mask = self.segconv(d1)  # 1×1卷积将厚度转为1
        mask = torch.sigmoid(mask)  # 归一化：转化为0-1的概率值
        mask_binary = (mask > 0.5)  # 二值化：概率大于 0.5 的设为 1（肿瘤），否则为0
        mask_binary = mask_binary.float()  # 这就是伪标签了。这两行可以合并成一行mask_binary = (mask > 0.5).float


        """这一段重复计算mask的解码似乎没有什么用并且会覆盖前面的所以注释掉
        # inpe1, inpe2, inpe3, inpe4, inpe5 = self.encoder(x)
        e1, e2, e3, e4, e5 = self.encoder(x)

        # 引入不确定性，用dropout扰动e5
        inpe1, inpe2, inpe3, inpe4, inpe5 = e1, e2, e3, e4, self.dropout(e5)
        
        d5 = self.segDecoder5(e5)
        d4 = self.segDecoder4(cat(d5, e4))
        d3 = self.segDecoder3(cat(d4, e3))
        d2 = self.segDecoder2(cat(d3, e2))
        d1 = self.segDecoder1(cat(d2, e1))
        
        mask = self.segconv(d1)
        mask = torch.sigmoid(mask)
        """
        # 引入不确定性，用dropout扰动e5
        inpe1, inpe2, inpe3, inpe4, inpe5 = e1, e2, e3, e4, self.dropout(e5)
        
        inpd5 = self.inpDecoder5(inpe5)
        inpimg5 = self.inpSideout5(inpd5)  # 从最深层输出一个小尺寸重建图Sideout
        inpimg5 = torch.sigmoid(inpimg5)
        inpd4 = self.inpDecoder4(cat(inpd5, inpe4))
        inpimg4 = self.inpSideout4(inpd4)
        inpimg4 = torch.sigmoid(inpimg4)
        inpd3 = self.inpDecoder3(cat(inpd4, inpe3))
        inpimg3 = self.inpSideout3(inpd3)
        inpimg3 = torch.sigmoid(inpimg3)
        inpd2 = self.inpDecoder2(cat(inpd3, inpe2))
        inpimg2 = self.inpSideout2(inpd2)
        inpimg2 = torch.sigmoid(inpimg2)
        inpd1 = self.inpDecoder1(cat(inpd2, inpe1))  # 最终层：预测物体的边界
        preboud = self.inpconv(inpd1)
        preboud = torch.sigmoid(preboud)

        # 返回 7 个变量。在训练脚本中，这些变量会被分别拿去算 Loss
        return mask, preboud, inpimg2, inpimg3, inpimg4, inpimg5, mask_binary
        # return mask, preboud, mask_binary

