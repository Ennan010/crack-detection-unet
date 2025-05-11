import torch
import torch.nn as nn
from dataset import AttentionGate

"""
U-Net模型实现，增加了注意力机制以提高裂缝检测精度。
本模型是标准U-Net的改进版本，通过在解码器部分添加注意力门控模块，
使网络能够更好地关注裂缝相关特征，抑制背景干扰。

U-Net model implementation with attention mechanism to improve crack detection accuracy.
This model is an improved version of the standard U-Net, by adding attention gate modules 
to the decoder part, making the network better at focusing on crack-related features 
while suppressing background interference.
"""

class DoubleConv(nn.Module):
    """
    双重卷积模块：U-Net的基本构建块
    包含两个连续的3×3卷积层，每个卷积后接BatchNorm和ReLU激活
    这种组合可以增强特征提取能力并加速训练过程
    
    Double Convolution Module: Basic building block of U-Net
    Contains two consecutive 3×3 convolution layers, each followed by BatchNorm and ReLU activation
    This combination enhances feature extraction capability and accelerates the training process
    
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
    
    Improved U-Net network for crack detection
    
    Main improvements:
    1. Adding attention gates (AttentionGate) to each decoder layer
    2. Using BatchNorm to accelerate training and improve stability
    
    参数 | Parameters:
        in_channels: 输入图像通道数，默认为3(RGB图像) | Input image channels, default is 3 (RGB image)
        out_channels: 输出通道数，默认为1(二元分割掩码) | Output channels, default is 1 (binary segmentation mask)
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # 编码器路径：连续的下采样过程，每层特征通道数翻倍
        # Encoder path: Continuous downsampling process, feature channels double at each layer
        self.enc1 = DoubleConv(in_channels, 64)  # 第一层编码器 | First encoder layer
        self.enc2 = DoubleConv(64, 128)          # 第二层编码器 | Second encoder layer
        self.enc3 = DoubleConv(128, 256)         # 第三层编码器 | Third encoder layer
        self.enc4 = DoubleConv(256, 512)         # 第四层编码器 | Fourth encoder layer
        
        # 网络最深层的瓶颈部分，具有最大的特征通道数
        # Bottleneck part at the deepest layer of the network, with the maximum number of feature channels
        self.bottleneck = DoubleConv(512, 1024)
        
        # 解码器路径：结合转置卷积上采样和注意力机制
        # 每层包括:
        # 1. 转置卷积上采样
        # 2. 注意力门控，增强相关特征
        # 3. 特征拼接
        # 4. 双重卷积处理
        
        # Decoder path: Combining transposed convolution upsampling and attention mechanism
        # Each layer includes:
        # 1. Transposed convolution upsampling
        # 2. Attention gate, enhancing relevant features
        # 3. Feature concatenation
        # 4. Double convolution processing
        
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
        # Final output layer: 1x1 convolution to map feature maps to required segmentation mask
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # 下采样操作：使用最大池化
        # Downsampling operation: using max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        前向传播过程:
        1. 编码器路径提取多尺度特征
        2. 瓶颈层捕获全局上下文
        3. 解码器路径结合注意力机制恢复空间细节
        
        Forward propagation process:
        1. Encoder path extracts multi-scale features
        2. Bottleneck layer captures global context
        3. Decoder path combines attention mechanism to restore spatial details
        """
        # 编码路径: 提取多尺度特征
        # Encoding path: Extract multi-scale features
        enc1 = self.enc1(x)                    # 最高分辨率特征 | Highest resolution features
        enc2 = self.enc2(self.pool(enc1))      # 第一次下采样 | First downsampling
        enc3 = self.enc3(self.pool(enc2))      # 第二次下采样 | Second downsampling
        enc4 = self.enc4(self.pool(enc3))      # 第三次下采样 | Third downsampling
        
        # 瓶颈: 最低分辨率，最高通道数
        # Bottleneck: Lowest resolution, highest number of channels
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # 解码路径: 结合注意力机制的特征融合
        # 处理最深层特征
        # Decoding path: Feature fusion with attention mechanism
        # Processing deepest layer features
        dec4 = self.up4(bottleneck)            # 上采样瓶颈特征 | Upsampling bottleneck features
        enc4_att = self.att4(dec4, enc4)       # 注意力处理编码器特征 | Attention processing encoder features
        dec4 = torch.cat((dec4, enc4_att), dim=1)  # 拼接特征 | Concatenating features
        dec4 = self.dec4(dec4)                 # 处理拼接后的特征 | Processing concatenated features
        
        # 处理第三层特征
        # Processing third layer features
        dec3 = self.up3(dec4)
        enc3_att = self.att3(dec3, enc3)
        dec3 = torch.cat((dec3, enc3_att), dim=1)
        dec3 = self.dec3(dec3)
        
        # 处理第二层特征
        # Processing second layer features
        dec2 = self.up2(dec3)
        enc2_att = self.att2(dec2, enc2)
        dec2 = torch.cat((dec2, enc2_att), dim=1)
        dec2 = self.dec2(dec2)
        
        # 处理第一层特征(最浅层)
        # Processing first layer features (shallowest layer)
        dec1 = self.up1(dec2)
        enc1_att = self.att1(dec1, enc1)
        dec1 = torch.cat((dec1, enc1_att), dim=1)
        dec1 = self.dec1(dec1)
        
        # 最终输出: 生成裂缝分割掩码
        # 注意：需要在外部使用sigmoid函数将输出转换为概率
        # Final output: Generate crack segmentation mask
        # Note: sigmoid function needs to be applied externally to convert output to probabilities
        return self.final_conv(dec1) 