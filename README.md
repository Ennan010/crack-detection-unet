# 裂缝检测 U-Net 模型 (Crack Detection U-Net Model)

<div align="center">
[中文文档](#中文文档) | [English Documentation](#english-documentation)
</div>
## Results

![2_prediction](D:\Code\Python\Crack_unet\crack_unet\results\2_prediction.png)

![5_prediction](D:\Code\Python\Crack_unet\crack_unet\results\5_prediction.png)



## 中文文档

### 项目简介

这个项目基于改进的 U-Net 网络结构实现了裂缝检测（Crack Detection）功能。模型可以从输入图像中识别并分割出裂缝区域，广泛适用于路面、墙体、桥梁等基础设施的裂缝检测场景。

### 模型设计改进

本项目不使用标准 U-Net 网络进行裂缝检测，而是通过引入注意力机制对网络结构进行了改进，显著提升了裂缝检测的精度：

1. **注意力机制增强** (Attention Gate)：在解码器的每一层添加注意力门控模块，使网络能够自适应地关注裂缝关键特征，同时抑制背景干扰。

2. **特征提取路径优化**：加强上下文信息与局部特征的融合，提高对细微裂缝的识别能力。

尽管加入了这些改进，模型在某些复杂场景（如纹理丰富的背景、光照不均匀区域、极细微裂缝等）下的表现仍有提升空间。当前版本在 CRACK500 数据集上取得了较好的性能，但实际应用中可能需要针对特定场景进行进一步优化。

### 项目结构

```
├── unet_model.py    # 改进型 U-Net 模型结构定义
├── dataset.py       # 数据集加载和预处理，包含注意力门控模块
├── train.py         # 模型训练脚本
├── predict.py       # 模型预测脚本
├── test_pic/        # 测试图像文件夹
├── results/         # 模型预测结果保存文件夹
├── output_results/  # 训练过程中模型和可视化结果保存文件夹
└── CRACK500/        # CRACK500 数据集文件夹
```

### 使用方法

#### 下载包 

```
pip install -r requirements.txt
```

#### 训练模型

```bash
python train.py
```

训练参数可以在 `train.py` 文件中调整，包括批次大小、学习率、训练轮数等。训练过程中会利用早停机制和学习率自动调节以避免过拟合。训练好的模型将保存在 `output_results` 文件夹中。

#### 预测单张图像

```bash
python predict.py --image test_pic/your_image.jpg --model output_results/best_model.pth --output results
```

参数说明：

- `--image`: 输入图像路径
- `--model`: 模型权重文件路径（默认为 `output_results/best_model.pth`）
- `--output`: 输出结果保存文件夹（默认为 `results`）
- `--threshold`: 二值化阈值，范围 0-1（默认为 0.5）
- `--no-postprocessing`: 禁用后处理操作（默认启用）

#### 预测全部测试图片示例命令

如果您需要单独处理 test_pic 文件夹中的特定测试图片，可以使用以下命令：

```bash
# 测试图片预测命令
python predict.py --image test_pic/1.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/2.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/3.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/4.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/5.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/road.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/road2.png --model output_results/best_model.pth --output results
```

### 预测结果格式

对于每个输入图像，预测结果将保存在 `results` 文件夹中，包括：

1. `{图像名称}_prediction.png`：三联图（原始图像、预测概率图、叠加显示）
2. `{图像名称}_mask.png`：二值化掩码图像，白色表示裂缝区域

---

## English Documentation

### Project Overview

This project implements crack detection functionality based on an improved U-Net network architecture. The model can identify and segment crack areas from input images, widely applicable to crack detection scenarios in infrastructure such as roads, walls, and bridges.

### Model Design Improvements

This project does not use the standard U-Net network for crack detection, but instead improves the network structure by introducing attention mechanisms, significantly enhancing crack detection accuracy:

1. **Attention Mechanism Enhancement** (Attention Gate): Adding attention gate modules to each layer of the decoder, enabling the network to adaptively focus on key crack features while suppressing background interference.

2. **Feature Extraction Path Optimization**: Strengthening the fusion of contextual information and local features to improve the recognition of fine cracks.

Despite these improvements, the model's performance still has room for enhancement in certain complex scenarios (such as richly textured backgrounds, unevenly lit areas, extremely fine cracks, etc.). The current version achieves good performance on the CRACK500 dataset, but may require further optimization for specific scenarios in practical applications.

### Project Structure

```
├── unet_model.py    # Improved U-Net model structure definition
├── dataset.py       # Dataset loading and preprocessing, including attention gate module
├── train.py         # Model training script
├── predict.py       # Model prediction script
├── test_pic/        # Test image folder
├── results/         # Model prediction results storage folder
├── output_results/  # Model and visualization results storage folder during training
└── CRACK500/        # CRACK500 dataset folder
```

### Usage

#### Package Installation

```
pip install -r requirements.txt
```

#### Training the Model

```bash
python train.py
```

Training parameters can be adjusted in the `train.py` file, including batch size, learning rate, number of epochs, etc. During training, early stopping mechanisms and automatic learning rate adjustment are used to avoid overfitting. The trained model will be saved in the `output_results` folder.

#### Predicting a Single Image

```bash
python predict.py --image test_pic/your_image.jpg --model output_results/best_model.pth --output results
```

Parameter description:

- `--image`: Input image path
- `--model`: Model weight file path (default: `output_results/best_model.pth`)
- `--output`: Output results save folder (default: `results`)
- `--threshold`: Binarization threshold, range 0-1 (default: 0.5)
- `--no-postprocessing`: Disable post-processing operations (enabled by default)

#### Example Commands for Predicting All Test Images

If you need to process specific test images in the test_pic folder individually, you can use the following commands:

```bash
# Test image prediction commands
python predict.py --image test_pic/1.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/2.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/3.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/4.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/5.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/road.jpg --model output_results/best_model.pth --output results
python predict.py --image test_pic/road2.png --model output_results/best_model.pth --output results
```

### Prediction Result Format

For each input image, the prediction results will be saved in the `results` folder, including:

1. `{image_name}_prediction.png`: Tri-view image (original image, prediction probability map, overlay display)
2. `{image_name}_mask.png`: Binarized mask image, white represents crack areas

