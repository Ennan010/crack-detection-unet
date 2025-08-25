import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import numpy as np
import random
import torch.nn as nn

"""
数据处理模块: 实现裂缝数据集的加载、预处理和增强
以及注意力门控模块的定义

"""

class CrackDataset(Dataset):
    """
    裂缝数据集加载类
    
    功能:
    1. 加载图像和对应的掩码
    2. 应用数据增强以提高模型泛化能力
    3. 提供统一的数据格式和预处理
    
    
    参数 | Parameters:
        image_dir: 原始图像所在目录 | Directory containing original images
        mask_dir: 掩码图像所在目录 | Directory containing mask images
        transform: 图像变换(通常用于调整大小和转为张量) | Image transformation (typically used for resizing and converting to tensors)
        augment: 是否启用数据增强 | Whether to enable data augmentation
    """
    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment
        
        # 检查目录是否存在 | Check if directories exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")
            
        # 获取所有图像文件名，只选择图像文件 | Get all image filenames, only select image files
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # 验证数据完整性：过滤掉没有对应掩码的图像 | Validate data integrity: filter out images without corresponding masks
        valid_images = []
        for img in self.images:
            base_name = os.path.splitext(img)[0]
            mask_path = os.path.join(mask_dir, base_name + "_mask.png")
            if os.path.exists(mask_path):
                valid_images.append(img)
            
        self.images = valid_images
        print(f"找到 {len(self.images)} 个有效的图像-掩码对 | Found {len(self.images)} valid image-mask pairs")
        
        # 打印前2个样本，帮助调试 | Print the first 2 samples to help debugging
        for i in range(min(2, len(self.images))):
            img_name = self.images[i]
            base_name = os.path.splitext(img_name)[0]
            mask_name = base_name + "_mask.png"
            print(f"图像 {i}: {img_name} | Image {i}: {img_name}")
            print(f"掩码 {i}: {mask_name} | Mask {i}: {mask_name}")
        
    def __len__(self):
        """返回数据集大小 | Return the size of the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        获取单个样本(图像和对应掩码)，并应用数据增强
        
        注意: 
        - 图像和掩码必须进行相同的几何变换以保持对齐
        - 只对图像进行亮度和对比度调整
        - 处理异常情况，确保训练过程不中断
        
        """
        try:
            # 加载图像 | Load image
            img_name = self.images[idx]
            img_path = os.path.join(self.image_dir, img_name)
            
            # 构建掩码路径 - 使用统一的命名规则 {base_name}_mask.png
            # Build mask path - using unified naming rule {base_name}_mask.png
            base_name = os.path.splitext(img_name)[0]
            mask_name = base_name + "_mask.png"
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            # 读取图像和掩码，并转换为合适的格式 | Read images and masks, and convert to appropriate format
            image = Image.open(img_path).convert('RGB')  # 确保3通道 | Ensure 3 channels
            mask = Image.open(mask_path).convert('L')    # 单通道灰度 | Single channel grayscale
            
            # 数据增强 (只在训练时随机应用) | Data augmentation (only randomly applied during training)
            if self.augment and random.random() > 0.5:
                # 1. 随机旋转 - 裂缝可能以任何角度出现
                # 1. Random rotation - cracks may appear at any angle
                angle = random.choice([90, 180, 270])
                image = image.rotate(angle)
                mask = mask.rotate(angle)  # 掩码也需要相同旋转 | Mask also needs the same rotation
                
                # 2. 随机水平翻转 | 2. Random horizontal flip
                if random.random() > 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                    
                # 3. 随机亮度、对比度调整(仅应用于图像，不应用于掩码)
                # 3. Random brightness and contrast adjustment (only applied to images, not to masks)
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))  # 亮度变化范围±20% | Brightness variation range ±20%
                
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))  # 对比度变化范围±20% | Contrast variation range ±20%
            
            # 应用其他变换(尺寸调整、转为张量等) | Apply other transformations (resize, convert to tensor, etc.)
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            
            # 确保掩码是二值的(0或1) | Ensure mask is binary (0 or 1)
            mask = (mask > 0.5).float()  # 大于0.5的像素视为裂缝(1)，否则为背景(0) | Pixels greater than 0.5 are considered cracks (1), otherwise background (0)
            
            return image, mask
            
        except Exception as e:
            # 错误处理：防止单个样本错误导致整个训练停止
            # Error handling: Prevent a single sample error from stopping the entire training
            print(f"处理索引 {idx} 的图像时出错: {e} | Error processing image at index {idx}: {e}")
            # 特殊情况：如果第一个样本就错误，创建零张量
            # Special case: If the first sample is already wrong, create zero tensors
            if idx == 0:
                image = torch.zeros((3, 256, 256))  # 创建空的RGB图像 | Create empty RGB image
                mask = torch.zeros((1, 256, 256))   # 创建空的掩码 | Create empty mask
                return image, mask
            # 否则尝试返回第一个样本 | Otherwise try to return the first sample
            return self.__getitem__(0)

class AttentionGate(nn.Module):
    """
    注意力门控模块
    
    这是本项目的核心改进部分，使网络能够关注裂缝区域并抑制背景噪声。
    工作原理：
    1. 分别处理来自上采样路径(g)和跳跃连接(x)的特征
    2. 计算特征间的相关性，生成注意力权重
    3. 用权重调整跳跃连接特征，突出重要区域
    
    参数 | Parameters:
        F_g: 上采样特征的通道数 | Number of channels for upsampling features
        F_l: 跳跃连接特征的通道数 | Number of channels for skip connection features
        F_int: 中间特征的通道数(降维用) | Number of channels for intermediate features (for dimensionality reduction)
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # a. 处理上采样特征的卷积层 | Convolution layer for processing upsampling features
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),  # 1x1卷积降维 | 1x1 convolution for dimensionality reduction
            nn.BatchNorm2d(F_int)                  # 批标准化提高稳定性 | Batch normalization for stability improvement
        )
        # b. 处理跳跃连接特征的卷积层 | Convolution layer for processing skip connection features
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),  # 1x1卷积降维 | 1x1 convolution for dimensionality reduction
            nn.BatchNorm2d(F_int)                  # 批标准化 | Batch normalization
        )
        # c. 生成注意力图的卷积层 | Convolution layer for generating attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),    # 输出单通道注意力图 | Output single channel attention map
            nn.BatchNorm2d(1),                     # 批标准化 | Batch normalization
            nn.Sigmoid()                           # 将值限制在0-1范围内 | Limit values to 0-1 range
        )
        # d. 激活函数 | Activation function
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        前向传播计算注意力权重并应用于特征
        
        参数:
            g: 上采样得到的特征(来自解码器)
            x: 跳跃连接特征(来自编码器)
            
        返回:
            x * psi: 加权后的跳跃连接特征
            
        """
        # 降维处理 | Dimensionality reduction processing
        g1 = self.W_g(g)      # 处理上采样特征 | Process upsampling features
        x1 = self.W_x(x)      # 处理跳跃连接特征 | Process skip connection features
        
        # 特征融合和激活 | Feature fusion and activation
        psi = self.relu(g1 + x1)  # 特征加和后ReLU激活 | ReLU activation after feature addition
        
        # 生成注意力系数(0-1范围) | Generate attention coefficients (0-1 range)
        psi = self.psi(psi)   # 生成注意力图 | Generate attention map
        
        # 注意力加权：相当于软掩码，突出重要区域 | Attention weighting: equivalent to soft masking, highlighting important areas
        return x * psi        # 将注意力应用到原始特征 | Apply attention to original features

