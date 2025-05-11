import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from unet_model import UNet
from dataset import CrackDataset
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只使用第二个GPU  
def dice_loss(pred, target):
    smooth = 1.0
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    # 早停参数
    patience = 15
    counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_steps = 0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        
        # 使用学习率调度器
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'当前学习率: {current_lr:.7f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0  # 重置早停计数器
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved to {model_path}!')
        else:
            counter += 1
            print(f'验证损失未改善。早停计数: {counter}/{patience}')
            if counter >= patience:
                print(f'早停! 连续{patience}个周期未改善验证损失')
                break
        
        # 可视化一些预测结果
        if (epoch + 1) % 5 == 0:
            visualize_predictions(model, val_loader, device, epoch, output_dir)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_curve_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path)
    print(f'Loss curve saved to {loss_curve_path}')
    plt.close()

def visualize_predictions(model, val_loader, device, epoch, output_dir):
    model.eval() # 设置为评估模式
    with torch.no_grad():
        # 获取一批验证数据
        images, masks = next(iter(val_loader))
        images = images.to(device)
        masks = masks.to(device)
        
        # 获取预测结果
        outputs = model(images)
        predictions = torch.sigmoid(outputs)
        
        # 选择前4个样本进行可视化
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        
        for i in range(4):
            # 原始图像
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # 真实标注
            mask = masks[i].cpu().numpy().squeeze()
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # 预测结果
            pred = predictions[i].cpu().numpy().squeeze()
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        pred_path = os.path.join(output_dir, f'predictions_epoch_{epoch+1}.png')
        plt.savefig(pred_path)
        print(f'Prediction visualization saved to {pred_path}')
        plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 输出目录
    output_dir = 'output_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')
    
    # 服务器路径
    images_path = 'CRACK500/CRACK500/JPEGImages/images'
    masks_path = 'CRACK500/CRACK500/Annotations/masks'
    
    # 数据预处理 
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 创建完整训练数据集
    full_dataset = CrackDataset(
        image_dir=images_path,
        mask_dir=masks_path,
        transform=transform,
        augment=True  # 启用数据增强
    )
    
    # 划分训练集和验证集（90%训练，10%验证）
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'验证集大小: {len(val_dataset)}')
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)
    
    # 创建模型
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # 定义损失函数和优化器
    criterion = lambda pred, target: 0.5 * nn.BCEWithLogitsLoss()(pred, target) + 0.5 * dice_loss(pred, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,  # 传入调度器
        num_epochs=80,
        device=device,
        output_dir=output_dir
    )

if __name__ == '__main__':
    main() 
