import os
import torch
import numpy as np
from PIL import Image, ImageFilter  # 使用PIL的滤镜功能
import matplotlib
# 设置无界面后端，避免Qt相关错误
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
from unet_model import UNet

def predict_image(model, image_path, device, threshold=0.5, output_dir="results", apply_postprocessing=True):
    """使用训练好的模型对单张图片进行裂缝检测"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    img_name = os.path.basename(image_path)
    base_name = os.path.splitext(img_name)[0]
    
    # 保存原始尺寸以便后续恢复
    original_size = img.size
    
    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # 应用阈值
    binary_mask = (pred > threshold).astype(np.uint8)
    
    # 后处理（如果启用）- 使用PIL过滤器
    if apply_postprocessing:
        # 将NumPy数组转为PIL图像进行处理
        mask_img = Image.fromarray(binary_mask * 255).convert('L')
        # 使用中值滤波去除小噪点
        mask_img = mask_img.filter(ImageFilter.MedianFilter(size=3))
        # 转回NumPy数组
        binary_mask = np.array(mask_img) > 127
    
    # 保存结果
    # 1. 原始图像
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.array(img))
    plt.title('Original Image')
    plt.axis('off')
    
    # 2. 预测掩码（原始概率图）
    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap='jet')  # 使用jet色彩映射更好地显示概率分布
    plt.colorbar(label='Crack Probability')
    plt.title(f'Prediction Mask (Threshold={threshold})')
    plt.axis('off')
    
    # 3. 叠加图 (将预测掩码叠加在原图上)
    plt.subplot(1, 3, 3)
    img_np = np.array(img.resize((256, 256))) / 255.0  # 调整为模型输出尺寸
    
    # 创建一个彩色掩码 (红色标记裂缝)
    colored_mask = np.zeros_like(img_np)
    colored_mask[..., 0] = binary_mask * 1.0  # 红色通道
    
    # 叠加图像 (70%原图 + 30%掩码)
    overlay = img_np * 0.7 + colored_mask * 0.3
    overlay = np.clip(overlay, 0, 1)
    
    plt.imshow(overlay)
    plt.title('Overlay Display')
    plt.axis('off')
    
    # 保存图像
    output_path = os.path.join(output_dir, f"{base_name}_prediction.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    # 保存二值化掩码（已后处理）
    binary_mask = binary_mask.astype(np.uint8) * 255
    mask_output = Image.fromarray(binary_mask).resize(original_size)
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    mask_output.save(mask_path)
    
    print(f"预测结果已保存到: {output_path}")
    print(f"二值化掩码已保存到: {mask_path}")
    return output_path

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用训练好的模型预测裂缝')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model', type=str, default='output_results/best_model.pth', help='模型权重文件')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值，范围0-1，默认0.5')
    parser.add_argument('--no-postprocessing', action='store_true', help='禁用后处理操作')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    print(f"已加载模型: {args.model}")
    
    # 预测图像
    predict_image(
        model, 
        args.image, 
        device, 
        threshold=args.threshold,
        output_dir=args.output,
        apply_postprocessing=not args.no_postprocessing
    )

if __name__ == '__main__':
    main() 