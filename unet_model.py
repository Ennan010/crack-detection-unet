import torch
import torch.nn as nn
from dataset import AttentionGate

"""
U-Net模型实现，增加了注意力机制以提高裂缝检测精度。
本模型是标准U-Net的改进版本，通过在解码器部分添加注意力门控模块，
使网络能够更好地关注裂缝相关特征，抑制背景干扰。

"""

class DoubleConv(nn.Module):
    """
    双重卷积模块：U-Net的基本构建块
    包含两个连续的3×3卷积层，每个卷积后接BatchNorm和ReLU激活
    这种组合可以增强特征提取能力并加速训练过程
    
    参数 | Parameters:
        in_channels: 输入通道数 | Input channels
        out_channels: 输出通道数 | Output channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    改进型U-Net网络，用于裂缝检测
    主要改进:
    1. 在每个解码器层添加注意力门控(AttentionGate)
    2. 使用BatchNorm加速训练并提高稳定性
    
    参数 | Parameters:
        in_channels: 输入图像通道数，默认为3(RGB图像) | Input image channels, default is 3 (RGB image)
        out_channels: 输出通道数，默认为1(二元分割掩码) | Output channels, default is 1 (binary segmentation mask)
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # 编码器路径：连续的下采样过程，每层特征通道数翻倍
       
        self.enc1 = DoubleConv(in_channels, 64)  # 第一层编码器 | First encoder layer
        self.enc2 = DoubleConv(64, 128)          # 第二层编码器 | Second encoder layer
        self.enc3 = DoubleConv(128, 256)         # 第三层编码器 | Third encoder layer
        self.enc4 = DoubleConv(256, 512)         # 第四层编码器 | Fourth encoder layer
        
        # 网络最深层的瓶颈部分，具有最大的特征通道数
        
        self.bottleneck = DoubleConv(512, 1024)
        
        # 解码器路径：结合转置卷积上采样和注意力机制
        # 每层包括:
        # 1. 转置卷积上采样
        # 2. 注意力门控，增强相关特征
        # 3. 特征拼接
        # 4. 双重卷积处理
        
        
        
        # 第一层解码器(最深层)
        # First decoder layer (deepest layer)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 上采样 | Upsampling
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)             # 注意力门控 | Attention gate
        self.dec4 = DoubleConv(1024, 512)                                  # 处理拼接后的特征 | Processing concatenated features
        
        # 第二层解码器
        # Second decoder layer
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = DoubleConv(512, 256)
        
        # 第三层解码器
        # Third decoder layer
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = DoubleConv(256, 128)
        
        # 第四层解码器(最浅层)
        # Fourth decoder layer (shallowest layer)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = DoubleConv(128, 64)
        
        # 最终输出层：1x1卷积将特征图映射为所需的分割掩码
       
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # 下采样操作：使用最大池化
       
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        前向传播过程:
        1. 编码器路径提取多尺度特征
        2. 瓶颈层捕获全局上下文
        3. 解码器路径结合注意力机制恢复空间细节
        
        """
        # 编码路径: 提取多尺度特征
        
        enc1 = self.enc1(x)                    # 最高分辨率特征 | Highest resolution features
        enc2 = self.enc2(self.pool(enc1))      # 第一次下采样 | First downsampling
        enc3 = self.enc3(self.pool(enc2))      # 第二次下采样 | Second downsampling
        enc4 = self.enc4(self.pool(enc3))      # 第三次下采样 | Third downsampling
        
        # 瓶颈: 最低分辨率，最高通道数
        # 解码路径: 结合注意力机制的特征融合
        # 处理最深层特征
        dec4 = self.up4(bottleneck)            # 上采样瓶颈特征 | Upsampling bottleneck features
        enc4_att = self.att4(dec4, enc4)       # 注意力处理编码器特征 | Attention processing encoder features
        dec4 = torch.cat((dec4, enc4_att), dim=1)  # 拼接特征 | Concatenating features
        dec4 = self.dec4(dec4)                 # 处理拼接后的特征 | Processing concatenated features
        
        # 处理第三层特征
        
        dec3 = self.up3(dec4)
        enc3_att = self.att3(dec3, enc3)
        dec3 = torch.cat((dec3, enc3_att), dim=1)
        dec3 = self.dec3(dec3)
        
        # 处理第二层特征
        
        dec2 = self.up2(dec3)
        enc2_att = self.att2(dec2, enc2)
        dec2 = torch.cat((dec2, enc2_att), dim=1)
        dec2 = self.dec2(dec2)
        
        # 处理第一层特征(最浅层)
        
        dec1 = self.up1(dec2)
        enc1_att = self.att1(dec1, enc1)
        dec1 = torch.cat((dec1, enc1_att), dim=1)
        dec1 = self.dec1(dec1)
        
        # 最终输出: 生成裂缝分割掩码
       
        return self.final_conv(dec1) 
