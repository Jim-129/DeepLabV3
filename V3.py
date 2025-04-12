from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import json
import argparse
import random
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import torch.nn.functional as F
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont

# 设置中文字体
def set_chinese_font():
    try:
        # 尝试使用不同中文字体
        font_options = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
        for font in font_options:
            try:
                plt.rcParams['font.family'] = [font, 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
                test_font = FontProperties(family=font)
                # 测试一下字体是否可用
                if test_font.get_name():  
                    print(f"成功设置中文字体: {font}")
                    return
            except:
                continue
        
        # 如果没有可用的中文字体，使用英文标签代替
        print("警告: 没有找到可用的中文字体，将使用英文标签")
    except Exception as e:
        print(f"设置中文字体时出错: {e}")

# 调用中文字体设置函数
set_chinese_font()

# 动态加载标签和颜色映射
def load_label_and_color_maps(sample_folder=None):
    """
    从两个可能的来源加载标签和颜色映射:
    1. 样本文件夹内的label_color_map.json文件
    2. Chair.txt文件
    
    Args:
        sample_folder: 样本文件夹路径(可选)
    
    Returns:
        label_map: 标签ID到名称的映射
        color_map: 标签ID到颜色的映射
    """
    label_map = {}
    color_map = {}
    
    # 默认颜色映射(备用)，仅包含背景和未使用类别
    default_color_map = {
        0: [128, 128, 128],  # 背景 - 灰色
        1: [192, 192, 192],  # 未使用 - 浅灰色
    }
    
    # 默认标签映射(备用)
    default_label_map = {
        0: "background",
        1: "unused",
    }
    
    # 1. 首先尝试从样本文件夹中的JSON文件加载
    if sample_folder and os.path.exists(sample_folder):
        json_path = os.path.join(sample_folder, "label_color_map.json")
        if os.path.exists(json_path):
            try:
                # print(f"正在从JSON文件加载映射: {json_path}")
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    
                if "labels" in json_data and "colors" in json_data:
                    # 直接使用JSON中的映射
                    for label_id_str, label_name in json_data["labels"].items():
                        label_id = int(label_id_str)
                        label_map[label_id] = label_name
                    
                    for label_id_str, color in json_data["colors"].items():
                        label_id = int(label_id_str)
                        color_map[label_id] = color
                    
                    # print(f"从JSON加载了 {len(label_map)} 个标签和 {len(color_map)} 个颜色映射")
                    return label_map, color_map
                else:
                    # 尝试直接解析JSON结构
                    for label_id_str, label_info in json_data.items():
                        if isinstance(label_info, list):
                            # 直接是颜色值
                            label_id = int(label_id_str)
                            color_map[label_id] = label_info
                        elif isinstance(label_info, str):
                            # 是标签名称
                            label_id = int(label_id_str)
                            label_map[label_id] = label_info
                    
                    if label_map and color_map:
                        print(f"从简化JSON加载了 {len(label_map)} 个标签和 {len(color_map)} 个颜色映射")
                        return label_map, color_map
            except Exception as e:
                print(f"从JSON加载映射时出错: {e}")
    
    # 3. 尝试加载失败时，使用硬编码的椅子部件标签而不是默认映射
    if not label_map or len(label_map) < 3:
        print("使用硬编码的椅子部件标签映射")
        label_map = {
            0: "背景",
            1: "裱花",
            2: "边框",
            3: "扶手",
            4: "侧立水",
            5: "前立水",
            6: "椅腿",
            7: "靠背",
            8: "坐垫",
            9: "枕头"
        }
        
        # 使用鲜明的颜色区分不同部件
        color_map = {
            0: [128, 128, 128],  # 背景 - 灰色
            1: [230, 25, 75],    # 裱花 - 红色
            2: [60, 180, 75],    # 边框 - 绿色
            3: [255, 225, 25],   # 扶手 - 黄色
            4: [0, 130, 200],    # 侧立水 - 蓝色
            5: [245, 130, 48],   # 前立水 - 橙色
            6: [145, 30, 180],   # 椅腿 - 紫色
            7: [70, 240, 240],   # 靠背 - 青色
            8: [240, 50, 230],   # 坐垫 - 品红
            9: [210, 245, 60],   # 枕头 - 酸橙色
        }
    
    return label_map, color_map

# 加载 DeepLabV3 模型
model = models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')

# 在PartNetSegDataset类定义前添加
def get_training_augmentation(strong_aug=True):
    """获取训练数据增强，添加强度参数"""
    # 基本变换
    train_transform = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.OneOf([
            A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=1.0),
            A.Compose([
                A.RandomScale(scale_limit=0.15, p=1.0),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
                A.PadIfNeeded(min_height=512, min_width=512, p=1.0),
                A.CenterCrop(height=512, width=512, p=1.0),
            ], p=1.0)
        ], p=1.0),
        A.RandomBrightnessContrast(p=0.2),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    ]
    
    # 如果需要强增强，添加更多变换
    if strong_aug:
        train_transform.extend([
            # 添加更多颜色变换
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.7),
                A.CLAHE(clip_limit=2.0, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5)
            ], p=0.3),
            
            # 添加噪声和模糊 - 增强模型鲁棒性
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5)
            ], p=0.2),
            
            # 网格和像素级扭曲 - 增强几何多样性
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.05, p=0.5)
            ], p=0.2),
        ])
    
    # 添加必要的归一化
    train_transform.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return A.Compose(train_transform)

def get_validation_augmentation():
    # 验证集只需要标准化处理
    test_transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(test_transform)

# 自定义数据集类，用于从生成的掩码中加载数据
class PartNetSegDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_training=False):
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        
        # 设置数据目录
        images_dir = r"D:\JimTemp\renet50\data\images"
        masks_dir = r"D:\JimTemp\renet50\data\masks"

        # 获取所有图像文件
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # 初始化变量
        self.samples = []
        all_label_maps = {}
        all_color_maps = {}
        max_class_id = 0
        first_sample_folder = None

        # 加载标签和颜色映射
        label_info_path = r"D:\JimTemp\renet50\data\label_color_map.json"
        if os.path.exists(label_info_path):
            with open(label_info_path, 'r') as f:
                label_info = json.load(f)
            
            # 转换标签映射（字符串键 -> 整数键）
            all_label_maps = {int(k): v for k, v in label_info.get('labels', {}).items()}
            all_color_maps = {int(k): v for k, v in label_info.get('colors', {}).items()}
            
            # 计算最大类别ID
            max_class_id = max(all_label_maps.keys()) if all_label_maps else 0
        else:
            # 如果没有找到标签信息文件，使用简单的二值映射
            print("警告: 未找到标签信息文件，使用二值映射")
            all_label_maps = {1: "对象"}
            all_color_maps = {1: (255, 255, 255)}
            max_class_id = 1

        # 保存所有标签和颜色映射
        self.label_map = all_label_maps
        self.color_map = all_color_maps
        self.num_classes = max_class_id + 1  # 类别数(包括背景)

        print(f"加载了 {len(self.samples)} 个样本，共 {self.num_classes} 个类别")
        print("\n类别信息:")
        for class_id, class_name in sorted(self.label_map.items()):
            print(f"  - ID {class_id}: {class_name}")

        # 添加额外的增强选项
        if self.is_training:
            self.strong_augmentation = get_training_augmentation()

        # 遍历所有图像文件，查找对应的掩码文件
        for img_file in image_files:
            # 构建完整图像路径
            img_path = os.path.join(images_dir, img_file)
            
            # 从图像文件名构建对应的掩码文件名
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}_mask.png"  # 根据您的实际命名规则进行调整
            mask_path = os.path.join(masks_dir, mask_file)
            
            # 检查掩码文件是否存在
            if not os.path.exists(mask_path):
                # 尝试其他可能的命名规则
                mask_files = [f for f in os.listdir(masks_dir) if f.startswith(base_name)]
                if mask_files:
                    mask_path = os.path.join(masks_dir, mask_files[0])
                else:
                    print(f"警告: 未找到图像 {img_file} 对应的掩码文件")
                    continue
            
            # 添加到样本列表
            self.samples.append({
                'rgb_img_path': img_path,
                'mask_path': mask_path,
                'folder': base_name  # 使用基础文件名作为文件夹名
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取图像和掩码
        img = cv2.imread(sample['rgb_img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        # 限制掩码值不超过类别数
        mask = np.clip(mask, 0, self.num_classes - 1)
        
        # 确保所有图像尺寸一致为512x512
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # 使用类别感知裁剪关注稀有类别
        if self.is_training and np.random.random() < 0.5:  # 50%的概率应用
            # 根据数据集中的稀有类别调整rare_class_ids参数
            img, mask = class_aware_crop(
                img, mask, 
                rare_class_ids=[2, 3, 4, 5],  # 稀有类别ID
                crop_size=(512, 512),  # 保持与输入大小一致
                p=0.9  # 高概率应用
            )
        
        # 应用数据增强
        if self.is_training and self.strong_augmentation is not None:
            augmented = self.strong_augmentation(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
            
            # 确保增强后图像尺寸仍然是512x512
            if isinstance(img, torch.Tensor):
                # 如果已经是张量，使用插值调整大小
                if img.shape[1:] != (512, 512):
                    img = F.interpolate(img.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(512, 512), mode='nearest').squeeze(0).squeeze(0).long()
            else:
                # 如果仍是numpy数组，使用cv2调整大小
                if img.shape[:2] != (512, 512):
                    img = cv2.resize(img, (512, 512))
                    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            
            # 确保mask是long类型
            if isinstance(mask, torch.Tensor):
                mask = mask.clone().detach().to(dtype=torch.long)
            else:
                mask = torch.tensor(mask, dtype=torch.long)
        elif self.transform is not None:
            img = self.transform(img)
            # 调整掩码大小与图像相同
            if img.shape[1:] != mask.shape:
                mask = cv2.resize(mask, (img.shape[2], img.shape[1]), interpolation=cv2.INTER_NEAREST)
            # 明确转换为long类型
            if isinstance(mask, torch.Tensor):
                mask = mask.clone().detach().to(dtype=torch.long)
            else:
                mask = torch.tensor(mask, dtype=torch.long)

        return img, mask

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 修改类别权重以优化分割效果
def calculate_class_weights(dataset, importance_factors=None):
    """
    计算每个类别的权重，并基于重要性因子进行调整
    
    Args:
        dataset: 数据集对象
        importance_factors: 类别ID到重要性因子的映射字典
    """
    # 添加类型检查
    if importance_factors is not None and not isinstance(importance_factors, dict):
        print(f"警告: importance_factors 必须是字典类型，而不是 {type(importance_factors)}，将被忽略")
        importance_factors = None
        
    class_counts = np.zeros(dataset.num_classes, dtype=np.float32)
    
    print("计算类别权重...")
    for i in tqdm(range(len(dataset))):
        _, mask = dataset[i]
        mask_np = mask.numpy()
        for c in range(dataset.num_classes):
            class_counts[c] += np.sum(mask_np == c)
    
    # 避免除零错误
    class_counts = np.where(class_counts == 0, 1, class_counts)
    
    # 计算权重 - 反比于频率
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (dataset.num_classes * class_counts)
    
    # 归一化权重
    class_weights = class_weights / np.sum(class_weights) * dataset.num_classes
    
    # 应用重要性因子进行权重调整
    if importance_factors is not None:
        for class_id, factor in importance_factors.items():
            if 0 <= class_id < len(class_weights):
                class_weights[class_id] *= factor
        
        # 再次归一化，确保权重总和保持不变
        class_weights = class_weights / np.sum(class_weights) * dataset.num_classes
    
    return class_weights

# 在模型训练部分之前添加焦点损失实现
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 创建一个类别平衡采样器
class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, num_classes, num_samples_per_class):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.class_indices = self._build_class_indices()
        
    def _build_class_indices(self):
        class_indices = [[] for _ in range(self.num_classes)]
        
        print("构建类别索引...")
        for i in tqdm(range(len(self.dataset))):
            _, mask = self.dataset[i]
            mask_np = mask.numpy()
            classes_in_mask = np.unique(mask_np)
            for c in classes_in_mask:
                # 只收集含有少数类的样本
                if c > 0 and c < self.num_classes:  # 跳过背景类
                    class_indices[c].append(i)
        
        return class_indices
        
    def __iter__(self):
        batch_indices = []
        # 优先选择包含少数类的样本
        for c in range(1, self.num_classes):  # 跳过背景类
            if len(self.class_indices[c]) > 0:
                indices = np.random.choice(
                    self.class_indices[c], 
                    size=min(self.num_samples_per_class, len(self.class_indices[c])),
                    replace=False
                )
                batch_indices.extend(indices)
        
        # 随机填充剩余的样本
        remaining = len(self.dataset) - len(batch_indices)
        if remaining > 0:
            all_indices = set(range(len(self.dataset)))
            remaining_indices = list(all_indices - set(batch_indices))
            additional = np.random.choice(
                remaining_indices,
                size=min(remaining, len(remaining_indices)),
                replace=False
            )
            batch_indices.extend(additional)
        
        np.random.shuffle(batch_indices)
        return iter(batch_indices)
    
    def __len__(self):
        return len(self.dataset)

# 添加评估函数
def evaluate_model(model, dataloader, device, num_classes):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备(CPU/GPU)
        num_classes: 类别数量
        
    Returns:
        metrics: 包含各种评估指标的字典
    """
    model.eval()
    conf_matrix = np.zeros((num_classes, num_classes))
    total_pixels = 0
    correct_pixels = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="评估模型性能"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            
            # 确保输出尺寸与掩码一致
            if outputs.shape[2:] != masks.shape[1:]:
                masks = nn.functional.interpolate(
                    masks.unsqueeze(1).float(), 
                    size=outputs.shape[2:], 
                    mode='nearest'
                ).squeeze(1).long()
            
            preds = torch.argmax(outputs, dim=1)
            
            # 统计准确率
            total_pixels += masks.numel()
            correct_pixels += (preds == masks).sum().item()
            
            # 更新混淆矩阵
            for i in range(masks.size(0)):
                conf = confusion_matrix(
                    masks[i].cpu().numpy().flatten(),
                    preds[i].cpu().numpy().flatten(),
                    labels=range(num_classes)
                )
                conf_matrix += conf
    
    # 计算各类别的IoU
    iou_per_class = np.zeros(num_classes)
    for i in range(num_classes):
        # TP / (TP + FP + FN)
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        iou = tp / (tp + fp + fn + 1e-10)  # 添加小值防止除零
        iou_per_class[i] = iou
    
    # 计算总体指标
    accuracy = correct_pixels / total_pixels
    mean_iou = np.mean(iou_per_class)
    
    # 在计算完IoU后添加边界F1分数计算
    boundary_f1 = 0
    num_samples = 0
    
    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="计算边界F1分数")):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)['out']
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = targets.cpu().numpy()
            
            # 对每个样本计算边界F1分数
            for j in range(preds.shape[0]):
                pred = preds[j]
                mask = targets[j]
                boundary_f1 += boundary_f1_score(pred, mask)
                num_samples += 1
    
    # 计算平均边界F1分数
    avg_boundary_f1 = boundary_f1 / num_samples if num_samples > 0 else 0
    
    # 在返回的指标中添加边界F1分数
    metrics = {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'conf_matrix': conf_matrix,
        'boundary_f1': avg_boundary_f1  # 新增边界F1分数
    }
    
    # 打印边界评估结果
    print(f"边界F1分数: {avg_boundary_f1:.4f}")
    
    return metrics

# 添加结果可视化函数
def plot_confusion_matrix(conf_matrix, class_names, output_path):
    """绘制并保存混淆矩阵图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=False, fmt='g', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Segmentation Model Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def plot_iou_per_class(iou_values, class_names, output_path):
    """绘制并保存每个类别的IoU柱状图"""
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(iou_values)), iou_values)
    plt.xticks(range(len(iou_values)), class_names, rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('IoU')
    plt.title('IoU per Class')
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def config(config_file, output_dir, data_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查配置文件是否存在
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            all_configs = config_data["configs"]
            # 检查是否有交叉验证配置
            use_cv = config_data.get("use_cross_validation", False)
            num_folds = config_data.get("num_folds", 5)
        
        # 训练日志文件
        log_file = os.path.join(output_dir, f"training_log_{datetime.now().strftime('%m%d_%H%M')}.txt")
        
        # 创建结果汇总数据框
        summary_data = []
        
        # 如果配置了交叉验证，则执行交叉验证流程
        if use_cv:
            print(f"\n{'='*50}")
            print(f"执行 {num_folds} 折交叉验证")
            print(f"{'='*50}\n")
            
            # 记录开始时间
            with open(log_file, 'a') as log:
                log.write(f"\n{'-'*30}\n")
                log.write(f"开始交叉验证 ({num_folds} 折)\n")
                log.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            try:
                # 提取第一个配置作为交叉验证参数
                cv_config = all_configs[0] if all_configs else {}
                
                # 执行交叉验证
                cv_result = cross_validation(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    num_folds=num_folds,
                    num_epochs=cv_config.get("num_epochs"),
                    batch_size=cv_config.get("batch_size"),
                    lr=cv_config.get("lr"),
                    weight_decay=cv_config.get("weight_decay"),
                    patience=cv_config.get("patience"),
                    # 修改这一行，确保传递的是字典或None，而不是布尔值
                    importance_factors=cv_config.get("importance_factors", None) if cv_config.get("use_importance_factors", False) else None
                )
                
                # 记录交叉验证结果
                with open(log_file, 'a') as log:
                    log.write(f"交叉验证完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log.write(f"平均准确率: {cv_result['avg_accuracy']:.4f}\n")
                    log.write(f"平均IoU: {cv_result['avg_mean_iou']:.4f}\n")
                    log.write(f"最佳模型路径: {cv_result['best_model_path']}\n\n")
                
                # 添加到汇总数据
                summary_entry = {
                    "Configuration": "交叉验证",
                    "Folds": num_folds,
                    "Epochs": cv_config.get("num_epochs"),
                    "Batch Size": cv_config.get("batch_size"),
                    "Learning Rate": cv_config.get("lr"),
                    "Weight Decay": cv_config.get("weight_decay"),
                    "Accuracy": cv_result['avg_accuracy'],
                    "Mean IoU": cv_result['avg_mean_iou']
                }
                
                # 添加每个类别的IoU
                for i, iou in enumerate(cv_result['avg_iou_per_class']):
                    label_name = f"Class {i}"
                    if i in cv_result['class_names']:
                        label_name = cv_result['class_names'][i]
                    summary_entry[f"IoU_{label_name}"] = iou
                
                summary_data.append(summary_entry)
            
            except Exception as e:
                print(f"交叉验证失败: {e}")
                with open(log_file, 'a') as log:
                    log.write(f"交叉验证失败: {str(e)}\n")

        # 在返回之前，将交叉验证结果添加到summary_data
        if summary_data:
            # 创建DataFrame并保存为CSV
            df = pd.DataFrame(summary_data)
            cv_csv_path = os.path.join(output_dir, f"cv_summary_{datetime.now().strftime('%m%d_%H%M')}.csv")
            df.to_csv(cv_csv_path, index=False, encoding='utf-8-sig')
            
            print(f"\n交叉验证结果汇总已保存到 {cv_csv_path}")

# 添加Dice损失，对小类别更敏感
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.weight = weight
        
    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        inputs_flat = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets_f = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        targets_flat = targets_f.view(targets_f.shape[0], targets_f.shape[1], -1)
        
        intersection = (inputs_flat * targets_flat).sum(2)
        dice = (2. * intersection + smooth) / (inputs_flat.sum(2) + targets_flat.sum(2) + smooth)
        return 1 - dice.mean(1).mean()

# 修改CombinedLoss增加边界感知
class CombinedLoss(nn.Module):
    def __init__(self, weights=None, alpha=0.5, boundary_weight=2.0):
        super(CombinedLoss, self).__init__()
        self.weights = weights
        self.alpha = alpha
        self.boundary_weight = boundary_weight
        
        # 初始化Sobel算子用于边缘检测
        self.sobel_x = torch.nn.Parameter(torch.Tensor([[-1, 0, 1], 
                                                        [-2, 0, 2], 
                                                        [-1, 0, 1]]), requires_grad=False).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.nn.Parameter(torch.Tensor([[-1, -2, -1], 
                                                        [0, 0, 0], 
                                                        [1, 2, 1]]), requires_grad=False).unsqueeze(0).unsqueeze(0)
    
    def forward(self, inputs, targets):
        # 标准交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weights)
        
        # Dice损失
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # 将目标转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        # 计算Dice损失
        intersection = (inputs_softmax * targets_one_hot).sum(dim=(2, 3))
        union = inputs_softmax.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        # 添加平滑以避免除零错误
        dice_score = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1 - dice_score.mean()
        
        # 组合损失
        combined = self.alpha * ce_loss + (1 - self.alpha) * dice_loss
        
        # 增加边界感知
        if self.boundary_weight > 0:
            # 转换为one-hot编码，用于边界提取
            one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
            
            # 为每个类别单独检测边缘
            edge_maps = []
            for cls in range(one_hot.size(1)):
                # 提取单个类别通道
                cls_channel = one_hot[:, cls:cls+1, :, :]
                
                # 应用Sobel算子检测边缘
                sobel_x = self.sobel_x.to(cls_channel.device)
                sobel_y = self.sobel_y.to(cls_channel.device)
                
                edge_x = F.conv2d(cls_channel, sobel_x, padding=1)
                edge_y = F.conv2d(cls_channel, sobel_y, padding=1)
                edge_mag = torch.sqrt(edge_x**2 + edge_y**2)
                edge_maps.append(edge_mag)
            
            # 合并所有类别的边缘图
            all_edges = torch.cat(edge_maps, dim=1)
            # 确保使用float类型
            edge_mask = (all_edges > 0.5).float()
            
            # 增加边界像素权重 - 将这个值增加到2.0或更高
            weights = torch.ones_like(targets, dtype=torch.float32) + edge_mask.sum(dim=1) * self.boundary_weight * 4.0
        
        # 应用边界权重到交叉熵 - 确保使用float
        weighted_ce = F.cross_entropy(inputs, targets, reduction='none')
        weighted_ce = weighted_ce * weights.float()
        boundary_loss = weighted_ce.mean()
        
        # 增加边界损失的权重到0.9
        return combined + boundary_loss * 0.9

# 添加到主函数外部，全局范围内
def smooth_prediction(pred, num_classes=None, preserve_boundaries=True):
    """更强的边界保留平滑"""
    # 如果未指定类别数，使用预测中的最大值+1
    if num_classes is None:
        num_classes = np.max(pred) + 1
    
    result = np.zeros_like(pred)
    
    # 提取所有边界
    edges_all = np.zeros_like(pred, dtype=np.uint8)
    
    # 对每个类别提取边界
    for c in range(1, num_classes):
        mask = (pred == c).astype(np.uint8)
        if np.sum(mask) > 0:
            # 使用Canny检测器提取边界
            edges = cv2.Canny(mask*255, 50, 150)
            # 略微扩张边界
            edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            edges_all[edges_dilated > 0] = c
    
    # 对每个类别应用开闭运算
    for c in range(1, num_classes):
        mask = (pred == c).astype(np.uint8)
        if np.sum(mask) > 0:
            # 强力平滑
            kernel = np.ones((5, 5), np.uint8)  # 更大的内核
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # 应用结果但保留边界
            result[mask == 1] = c
    
    # 恢复边界
    for c in range(1, num_classes):
        result[edges_all == c] = c
    
    # 将未分类区域设为背景
    result[result == 0] = 0
    
    return result

def main(num_epochs, batch_size, lr, weight_decay, patience, data_dir, output_dir, importance_factors=None):
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description='DeepLabV3模型训练')
    parser.add_argument('--resume', action='store_true', help='从上次训练的模型继续训练')
    parser.add_argument('--model_path', type=str, default=None, help='要继续训练的模型路径')
    args = parser.parse_args()

    # 仅当importance_factors不为None时才进行处理
    if importance_factors:
        importance_factors = {int(k): float(v) for k, v in importance_factors.items()}
        # 打印实际使用的重要性因子
        print(f"使用的重要性因子: {importance_factors}")
    else:
        print("不使用特殊的重要性因子，仅使用基本类别平衡")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 从层次结构文件中加载类别信息
    hierarchy_labels, _ = load_label_and_color_maps()
    # 从层次结构文件中加载类别信息
    # hierarchy_labels, hierarchy_colors = load_label_and_color_maps()    
    
    # 加载数据集 - 移除 augmentations 参数
    full_dataset = PartNetSegDataset(data_dir, transform=transform, is_training=True)

    # 获取类别数
    # num_classes = max(full_dataset.num_classes, 58)
    num_classes = full_dataset.num_classes
    print(f"检测到 {num_classes} 个类别")
    
    # 结合两种方式的类别名称
    combined_class_names = full_dataset.label_map.copy()
    if hierarchy_labels:
        for label_id, label_name in hierarchy_labels.items():
            if 0 <= label_id < full_dataset.num_classes:
                if label_id not in combined_class_names or combined_class_names[label_id] == f"类别 {label_id}":
                    combined_class_names[label_id] = label_name
    
    # 打印综合的类别信息
    print("\n综合类别信息:")
    for class_id in range(full_dataset.num_classes):
        class_name = combined_class_names.get(class_id, f"类别 {class_id}")
        print(f"  - ID {class_id}: {class_name}")
    
    # 计算类别权重，传入重要性因子
    class_weights = calculate_class_weights(full_dataset, importance_factors if importance_factors else None)
    print("类别权重计算完成:", class_weights)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # 使用常规交叉熵损失与权重
    # criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    criterion = CombinedLoss(weights=class_weights_tensor, alpha=0.5, boundary_weight=2.0)
    
    # 加载预训练模型或初始化新模型
    if args.resume and (args.model_path is not None) and os.path.exists(args.model_path):
        print(f"从保存的模型继续训练: {args.model_path}")
        # 初始化模型并调整分类器输出层
        model = models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
        
        # 加载保存的模型状态
        state_dict = torch.load(args.model_path)
        
        # 检查分类器的输出通道是否匹配
        if 'classifier.4.weight' in state_dict:
            saved_out_channels = state_dict['classifier.4.weight'].size(0)
            if saved_out_channels != num_classes:
                print(f"警告: 保存的模型有 {saved_out_channels} 个类别，当前数据集有 {num_classes} 个类别")
                print("重新初始化分类器层...")
                # 移除不匹配的分类器权重
                state_dict.pop('classifier.4.weight', None)
                state_dict.pop('classifier.4.bias', None)
                
        # 加载模型权重
        try:
            model.load_state_dict(state_dict, strict=False)
            print("成功加载模型权重")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("将使用部分加载的模型继续训练")
    else:
        # 创建新模型
        model = models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
        # 确保模型分类器输出足够的类别
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
        print("初始化新模型")
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # 使用平衡采样器创建数据加载器
    train_sampler = BalancedBatchSampler(train_dataset, num_classes, num_samples_per_class=1)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,  # 使用自定义采样器
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True  # 丢弃最后一个不完整的批次
    )
       
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay,  # 直接使用配置中的权重衰减值
        betas=(0.9, 0.999),         # AdamW默认值
        eps=1e-8                     # AdamW默认值
    )

    # 学习率衰减调度器 - 余弦退火可以更好地优化边界
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=lr * 0.01  # 最小学习率为初始值的1%
    )

    counter = 0
    best_val_loss = float('inf')
    
    # 如果继续训练，检查是否有保存的最佳验证损失
    if args.resume and (args.model_path is not None) and os.path.exists(args.model_path):
        # 尝试从模型文件名中提取之前的验证损失（如果有）
        model_name = os.path.basename(args.model_path)
        try:
            if "val_loss" in model_name:
                loss_str = model_name.split("val_loss_")[1].split(".pth")[0]
                prev_val_loss = float(loss_str)
                best_val_loss = prev_val_loss
                print(f"从文件名恢复最佳验证损失: {best_val_loss:.4f}")
        except:
            pass
    
    print(f"开始训练，最佳验证损失初始值: {best_val_loss:.4f}")

    # 在创建优化器后添加学习曲线记录
    train_losses = []
    val_losses = []
    
    # 训练循环中记录损失
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_loop = tqdm(train_loader, desc=f"第 {epoch+1}/{num_epochs} 轮 [训练]")
        
        for images, masks in train_loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']
            
            # 确保输出尺寸与掩码一致
            if outputs.shape[2:] != masks.shape[1:]:
                masks = nn.functional.interpolate(
                    masks.unsqueeze(1).float(), 
                    size=outputs.shape[2:], 
                    mode='nearest'
                ).squeeze(1).long()
                
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        with torch.no_grad():
            model.eval()  # 设置为评估模式
            val_loss = 0
            val_loop = tqdm(val_loader, desc=f"第 {epoch+1}/{num_epochs} 轮 [验证]")
            for images, masks in val_loop:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                
                # 确保输出尺寸与掩码一致
                if outputs.shape[2:] != masks.shape[1:]:
                    masks = nn.functional.interpolate(
                        masks.unsqueeze(1).float(), 
                        size=outputs.shape[2:], 
                        mode='nearest'
                    ).squeeze(1).long()
                    
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())
            model.train()  # 完成验证后恢复训练模式
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"第 {epoch+1}/{num_epochs} 轮, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        
        # 记录损失值用于绘制学习曲线
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 早停法和模型保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 添加时间戳到模型文件名
            timestamp = datetime.now().strftime("%d_%H%M%S")
            model_save_path = os.path.join(output_dir, f"best_model_{timestamp}_val_loss_{best_val_loss:.4f}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存到 {model_save_path}，验证损失：{best_val_loss:.4f}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"在第 {epoch+1} 轮提前停止训练")
                break

    # 保存最终模型时也添加时间戳
    timestamp = datetime.now().strftime("%d_%H%M")
    torch.save(model.state_dict(), os.path.join(output_dir, f"segmentation_model_final_{timestamp}.pth"))
    
    # 测试和可视化部分
    model.eval()
    
    # 使用验证集的一部分作为测试样本
    test_indices = val_dataset.indices[-10:]  # 使用验证集的最后10个样本
    test_samples = [full_dataset.samples[i] for i in test_indices]
    
    # 处理测试样本
    for i, sample in enumerate(test_samples):
        print(f"处理测试图像 {i+1}/{len(test_samples)} - {sample['folder']}")
        
        # 从样本文件夹加载标签和颜色映射
        sample_folder = os.path.join(data_dir, sample['folder'])
        label_map, color_map_dict = load_label_and_color_maps(sample_folder)
        
        # 将颜色映射字典转换为NumPy数组
        max_label = max(max(color_map_dict.keys()) + 1, model.classifier[4].out_channels)
        COLOR_MAP = np.zeros((max_label, 3), dtype=np.uint8)
        
        # 为所有没有指定颜色的类别生成随机颜色
        for i in range(max_label):
            if i in color_map_dict:
                COLOR_MAP[i] = color_map_dict[i]
            else:
                # 为未指定颜色的类别生成随机颜色
                COLOR_MAP[i] = np.random.randint(0, 256, 3)
        
        # 读取图像和真实掩码
        img = cv2.imread(sample['rgb_img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        # 预处理并预测
        img_resized = cv2.resize(img, (512, 512))
        img_tensor = transform(img_resized).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)['out']
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # 应用平滑处理
        pred = smooth_prediction(pred, num_classes=model.classifier[4].out_channels)
        
        # 调整预测尺寸以匹配原始图像
        if pred.shape != (img.shape[0], img.shape[1]):
            pred = cv2.resize(pred.astype(np.uint8), (img.shape[1], img.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        # 创建彩色掩码 - 确保索引不超出COLOR_MAP的范围
        pred_safe = np.clip(pred, 0, max_label-1)
        mask_safe = np.clip(mask, 0, max_label-1)
        
        # 使用动态加载的颜色映射为掩码着色
        colored_pred = COLOR_MAP[pred_safe].astype(np.uint8)
        colored_true = COLOR_MAP[mask_safe].astype(np.uint8)
        
        # 创建叠加图
        alpha = 0.6
        overlay = cv2.addWeighted(img, 1-alpha, colored_pred, alpha, 0)
        
        # 为当前样本创建输出目录
        sample_output_dir = os.path.join(output_dir, f"timestamp_{timestamp}")
        if not os.path.exists(sample_output_dir):
            os.makedirs(sample_output_dir)
        
        # 保存结果
        cv2.imwrite(os.path.join(sample_output_dir, 'original.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(sample_output_dir, 'true_mask.png'), cv2.cvtColor(colored_true, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(sample_output_dir, 'pred_mask.png'), cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(sample_output_dir, 'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # 创建带图例的组合图像
        # 原图 + 预测 + 图例
        combined_img = np.zeros((max(img.shape[0], colored_pred.shape[0]), 
                                img.shape[1] + colored_pred.shape[1] + 300, 3), dtype=np.uint8)
        
        # 添加原图
        combined_img[0:img.shape[0], 0:img.shape[1]] = img
        # 添加预测掩码
        combined_img[0:colored_pred.shape[0], img.shape[1]:img.shape[1]+colored_pred.shape[1]] = colored_pred
        
        # 创建图例
        legend = create_legend_with_chinese(color_map_dict, label_map)
        
        # 添加图例 (居中放置)
        legend_y_offset = (combined_img.shape[0] - legend.shape[0]) // 2
        if legend_y_offset >= 0:
            combined_img[legend_y_offset:legend_y_offset+legend.shape[0], 
                       img.shape[1]+colored_pred.shape[1]:] = legend
        
        # 保存组合图像
        cv2.imwrite(os.path.join(sample_output_dir, 'combined.png'), 
                   cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        
        print(f"处理完成: {sample['folder']}")
    
    print("所有处理完成！")    

    # 在训练结束后添加评估代码
    print("模型训练完成，开始评估...")
    
    # 使用验证集评估模型
    metrics = evaluate_model(model, val_loader, device, num_classes)
    
    # 保存评估结果
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        # 将numpy数组转换为列表以便JSON序列化
        results_dict = {
            'accuracy': metrics['accuracy'],
            'mean_iou': metrics['mean_iou'],
            'iou_per_class': metrics['iou_per_class'].tolist(),
        }
        json.dump(results_dict, f, indent=4)
    
    # 生成可视化结果
    class_names = [combined_class_names.get(i, f"类别{i}") for i in range(num_classes)]
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        metrics['conf_matrix'], 
        class_names, 
        os.path.join(output_dir, "confusion_matrix.png")
    )
    
    # 绘制每个类别的IoU
    plot_iou_per_class(
        metrics['iou_per_class'],
        class_names,
        os.path.join(output_dir, "iou_per_class.png")
    )
    
    print(f"评估结果已保存到 {results_file}")
    print(f"准确率: {metrics['accuracy']:.4f}, 平均IoU: {metrics['mean_iou']:.4f}")
    
    # 添加学习曲线绘制
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))
    plt.close()
    
    # 在函数末尾返回更多信息
    return {
        'metrics': metrics,
        'class_names': combined_class_names
    }

# 完全重写边界F1计算方法，更准确地反映边界质量
def boundary_f1_score(pred, mask, class_wise=True, distance_thresh=3):
    """
    计算基于区域的边界评估指标，更符合实际感知质量
    
    Args:
        pred: 预测分割图
        mask: 真实分割图
        class_wise: 是否分类别计算
        distance_thresh: 边界容差阈值(像素)
    """
    # 初始化指标
    num_classes = len(np.unique(mask))
    
    # 获取每个类别的边界IoU
    class_scores = []
    
    # 对每个类别计算区域指标
    for c in np.unique(mask):
        if c == 0:  # 跳过背景
                continue
            
        # 提取类别二值掩码
        pred_mask = (pred == c).astype(np.uint8)
        gt_mask = (mask == c).astype(np.uint8)
        
        # 如果真实掩码中没有此类别，跳过
        if np.sum(gt_mask) == 0:
            continue
        
        # 计算区域指标(区域IoU)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / (union + 1e-7)
        
        # 获取区域边界
        gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果没有轮廓，跳过
        if len(gt_contours) == 0 or len(pred_contours) == 0:
            continue
            
        # 计算边界准确度
        # 创建边界图像
        gt_boundary = np.zeros_like(gt_mask)
        cv2.drawContours(gt_boundary, gt_contours, -1, 1, 1)
        
        pred_boundary = np.zeros_like(pred_mask)
        cv2.drawContours(pred_boundary, pred_contours, -1, 1, 1)
        
        # 使用距离变换计算边界匹配
        gt_dt = cv2.distanceTransform((1-gt_boundary).astype(np.uint8), cv2.DIST_L2, 3)
        pred_dt = cv2.distanceTransform((1-pred_boundary).astype(np.uint8), cv2.DIST_L2, 3)
        
        # 计算边界匹配
        gt_matched = (gt_dt <= distance_thresh) & (pred_boundary > 0)
        pred_matched = (pred_dt <= distance_thresh) & (gt_boundary > 0)
        
        # 计算边界F1
        precision = np.sum(gt_matched) / (np.sum(pred_boundary) + 1e-7)
        recall = np.sum(pred_matched) / (np.sum(gt_boundary) + 1e-7)
        
        boundary_f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        # 综合指标 = 0.7*IoU + 0.3*边界F1
        combined_score = 0.7 * iou + 0.3 * boundary_f1
        class_scores.append(combined_score)
    
    # 计算平均分数
    if len(class_scores) > 0:
        average_score = np.mean(class_scores)
    else:
        average_score = 0.0
    
    # 打印结果
    print(f"区域级边界评估: score={average_score:.4f}")
    
    return average_score

# 添加类别感知裁剪函数
def class_aware_crop(image, mask, rare_class_ids=[2, 3, 4, 5], crop_size=(256, 256), p=0.7):
    """
    对包含稀有类别的区域进行主动裁剪
    """
    if np.random.random() > p:
        return image, mask
    
    # 找出所有稀有类别的位置
    rare_pixels = np.zeros_like(mask, dtype=bool)
    for cls_id in rare_class_ids:
        rare_pixels = np.logical_or(rare_pixels, mask == cls_id)
    
    # 如果没有稀有类别，返回原始图像
    if not np.any(rare_pixels):
        return image, mask
    
    # 获取稀有类别的坐标
    y_indices, x_indices = np.where(rare_pixels)
    
    # 随机选择一个稀有类别像素作为裁剪中心
    idx = np.random.randint(0, len(y_indices))
    center_y, center_x = y_indices[idx], x_indices[idx]
    
    # 计算裁剪边界
    half_height, half_width = crop_size[0] // 2, crop_size[1] // 2
    start_y = max(0, center_y - half_height)
    start_x = max(0, center_x - half_width)
    end_y = min(mask.shape[0], center_y + half_height)
    end_x = min(mask.shape[1], center_x + half_width)
    
    # 裁剪图像和掩码
    cropped_image = image[start_y:end_y, start_x:end_x]
    cropped_mask = mask[start_y:end_y, start_x:end_x]
    
    # 调整到目标大小
    if cropped_image.shape[:2] != crop_size:
        cropped_image = cv2.resize(cropped_image, crop_size[::-1], interpolation=cv2.INTER_LINEAR)
        cropped_mask = cv2.resize(cropped_mask, crop_size[::-1], interpolation=cv2.INTER_NEAREST)
    
    return cropped_image, cropped_mask

# 使用PIL绘制支持中文的图例
def create_legend_with_chinese(color_map_dict, label_map, width=300, max_height=900):
    # 计算图例高度
    legend_height = min(30 * (len(color_map_dict) + 1), max_height)
    
    # 创建白色背景图像
    legend = Image.new('RGB', (width, legend_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(legend)
    
    # 尝试加载中文字体
    try:
        # 尝试几种常见的中文字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # Linux中的中文字体
        ]
        
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, 14)
                print(f"成功加载字体: {path}")
                break
            except:
                continue
        
        if font is None:
            # 如果找不到中文字体，使用默认字体
            font = ImageFont.load_default()
            print("警告：未找到中文字体，使用默认字体")
    except Exception as e:
        font = ImageFont.load_default()
        print(f"加载字体出错: {e}")
    
    # 绘制图例
    for idx, (class_idx, class_name) in enumerate(sorted(label_map.items())):
        y_start = idx * 30 + 5
        if y_start + 20 > legend_height:
            break
        
        # 绘制颜色方块
        if class_idx in color_map_dict:
            color = tuple(color_map_dict[class_idx])
            draw.rectangle([(10, y_start), (50, y_start+20)], fill=color, outline=(0,0,0))
            
            # 添加类别文本
            label_text = f"类别 {class_idx}: {class_name}"
            draw.text((60, y_start+5), label_text, fill=(0,0,0), font=font)
    
    # 添加未在label_map中但在color_map中的类别
    extra_idx = 0
    for class_idx, color in color_map_dict.items():
        if class_idx not in label_map:
            y_start = len(label_map) * 30 + extra_idx * 30 + 5
            if y_start + 20 > legend_height:
                break
            
            # 绘制颜色方块
            draw.rectangle([(10, y_start), (50, y_start+20)], fill=tuple(color), outline=(0,0,0))
            
            # 添加类别文本
            label_text = f"未知类别 {class_idx}"
            draw.text((60, y_start+5), label_text, fill=(0,0,0), font=font)
            extra_idx += 1
    
    # 转换为numpy数组以与OpenCV兼容
    return np.array(legend)

# 添加自定义的多尺度特征融合模型
class MultiScaleDeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(MultiScaleDeepLabV3, self).__init__()
        # 使用预训练的DeepLabV3
        self.deeplab = models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
        
        # 修改分类器以适应新的类别数
        self.deeplab.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # 访问基础resnet的底层特征
        self.backbone = self.deeplab.backbone
        
        # 多尺度特征提取器 - 从不同层获取特征
        # layer1 - 低级特征，layer4 - 高级特征
        self.low_level_features = nn.Sequential(
            nn.Conv2d(256, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # 融合高低级特征
        self.fusion = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # 特征金字塔池化模块 - 捕获多尺度上下文
        self.aspp = self.deeplab.classifier[0]
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # 获取backbone特征
        features = self.backbone(x)
        
        # 低级特征 (layer1)
        low_level_feat = features['out']
        low_level_feat = self.low_level_features(low_level_feat)
        
        # 高级特征 (通过ASPP)
        high_level_feat = self.aspp(features['out'])
        
        # 上采样高级特征
        high_level_feat = F.interpolate(high_level_feat, size=low_level_feat.shape[2:], 
                                       mode='bilinear', align_corners=True)
        
        # 特征融合
        fused_features = torch.cat([high_level_feat, low_level_feat], dim=1)
        output = self.fusion(fused_features)
        
        # 上采样到原始尺寸
        output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=True)
        
        return {'out': output}

# 恢复标准的DeepLabV3模型，但保留优化器改进
def initialize_model(num_classes, device, use_multiscale=False):
    """初始化分割模型"""
    if use_multiscale:
        # 使用多尺度模型
        model = MultiScaleDeepLabV3(num_classes)
    else:
        # 使用标准DeepLabV3
        model = models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
        # 修改分类器头以适应类别数
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    model = model.to(device)
    return model

# 在import部分添加scikit-learn的KFold
from sklearn.model_selection import KFold

# 在main函数前添加交叉验证函数
def cross_validation(data_dir, output_dir, num_folds, num_epochs, batch_size, 
                    lr, weight_decay, patience, importance_factors=None):
    # 函数开始时添加参数检查
    if importance_factors is not None and not isinstance(importance_factors, dict):
        print(f"警告: importance_factors 参数类型错误，期望字典类型，实际为 {type(importance_factors)}")
        importance_factors = None  # 设置为 None 避免后续错误
    
    print(f"\n{'='*50}")
    print(f"开始 {num_folds} 折交叉验证")
    print(f"{'='*50}\n")
    
    # 创建交叉验证输出目录
    cv_output_dir = os.path.join(output_dir, "cross_validation")
    os.makedirs(cv_output_dir, exist_ok=True)
    
    # 加载完整数据集
    full_dataset = PartNetSegDataset(data_dir, transform=transform, is_training=True)
    num_classes = full_dataset.num_classes
    print(f"数据集共有 {len(full_dataset)} 个样本，{num_classes} 个类别")
    
    # 获取类别名称映射
    hierarchy_labels, _ = load_label_and_color_maps()
    combined_class_names = full_dataset.label_map.copy()
    if hierarchy_labels:
        for label_id, label_name in hierarchy_labels.items():
            if 0 <= label_id < full_dataset.num_classes:
                if label_id not in combined_class_names or combined_class_names[label_id] == f"类别 {label_id}":
                    combined_class_names[label_id] = label_name
    
    # 创建K折交叉验证分割器
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # 存储每折的结果
    fold_results = []
    best_models = []
    all_val_metrics = []
    
    # 生成所有样本的索引
    indices = np.arange(len(full_dataset))
    
    # 对每一折进行训练和评估
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n{'='*50}")
        print(f"开始第 {fold+1}/{num_folds} 折训练")
        print(f"训练集: {len(train_idx)} 样本, 验证集: {len(val_idx)} 样本")
        print(f"{'='*50}\n")
        
        # 创建当前折的输出目录
        fold_output_dir = os.path.join(cv_output_dir, f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # 创建训练集和验证集
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        
        # 获取设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        model = models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
        model.to(device)
        
        # 计算类别权重
        class_weights = calculate_class_weights(full_dataset, importance_factors)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        
        # 使用组合损失函数
        criterion = CombinedLoss(weights=class_weights_tensor, alpha=0.5, boundary_weight=2.0)
        
        # 使用平衡采样器创建数据加载器
        train_sampler = BalancedBatchSampler(train_dataset, num_classes, num_samples_per_class=1)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=True
        )
        
        # 初始化优化器
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs,
            eta_min=lr * 0.01
        )
        
        # 训练变量初始化
        best_val_loss = float('inf')
        counter = 0
        train_losses = []
        val_losses = []
        best_model_path = None
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_loop = tqdm(train_loader, desc=f"第 {epoch+1}/{num_epochs} 轮 [训练] (折 {fold+1}/{num_folds})")
            
            for images, masks in train_loop:
                images, masks = images.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)['out']
                
                # 确保输出尺寸与掩码一致
                if outputs.shape[2:] != masks.shape[1:]:
                    masks = nn.functional.interpolate(
                        masks.unsqueeze(1).float(), 
                        size=outputs.shape[2:], 
                        mode='nearest'
                    ).squeeze(1).long()
                    
                loss = criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_loop.set_postfix(loss=loss.item())
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            with torch.no_grad():
                model.eval()
                val_loss = 0
                val_loop = tqdm(val_loader, desc=f"第 {epoch+1}/{num_epochs} 轮 [验证] (折 {fold+1}/{num_folds})")
                for images, masks in val_loop:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)['out']
                    
                    # 确保输出尺寸与掩码一致
                    if outputs.shape[2:] != masks.shape[1:]:
                        masks = nn.functional.interpolate(
                            masks.unsqueeze(1).float(), 
                            size=outputs.shape[2:], 
                            mode='nearest'
                        ).squeeze(1).long()
                        
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    val_loop.set_postfix(loss=loss.item())
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"第 {epoch+1}/{num_epochs} 轮, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
            
            # 记录损失值用于绘制学习曲线
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # 更新学习率
            scheduler.step()
            
            # 早停法和模型保存
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                timestamp = datetime.now().strftime("%d_%H%M%S")
                best_model_path = os.path.join(fold_output_dir, f"best_model_fold{fold+1}_{timestamp}_val_loss_{best_val_loss:.4f}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"模型已保存到 {best_model_path}，验证损失：{best_val_loss:.4f}")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"在第 {epoch+1} 轮提前停止训练")
                    break
        
        # 绘制学习曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='训练损失')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title(f'第 {fold+1} 折学习曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fold_output_dir, f'learning_curve_fold{fold+1}.png'))
        plt.close()
        
        # 使用最佳模型进行评估
        if best_model_path:
            # 加载最佳模型
            model.load_state_dict(torch.load(best_model_path))
            model.eval()
            
            # 评估模型性能
            metrics = evaluate_model(model, val_loader, device, num_classes)
            
            # 保存评估结果
            results_file = os.path.join(fold_output_dir, "evaluation_results.json")
            with open(results_file, 'w') as f:
                results_dict = {
                    'accuracy': metrics['accuracy'],
                    'mean_iou': metrics['mean_iou'],
                    'iou_per_class': metrics['iou_per_class'].tolist(),
                    'fold': fold+1,
                    'best_val_loss': best_val_loss
                }
                json.dump(results_dict, f, indent=4)
            
            # 保存结果到列表
            fold_results.append({
                'fold': fold+1,
                'best_val_loss': best_val_loss,
                'accuracy': metrics['accuracy'],
                'mean_iou': metrics['mean_iou'],
                'best_model_path': best_model_path
            })
            
            best_models.append(best_model_path)
            all_val_metrics.append(metrics)
            
            # 绘制每个类别的IoU
            class_names = [combined_class_names.get(i, f"类别{i}") for i in range(num_classes)]
            plot_iou_per_class(
                metrics['iou_per_class'],
                class_names,
                os.path.join(fold_output_dir, f"iou_per_class_fold{fold+1}.png")
            )
            
            print(f"第 {fold+1} 折评估结果: 准确率={metrics['accuracy']:.4f}, 平均IoU={metrics['mean_iou']:.4f}")
    
    # 汇总所有折的结果
    print("\n交叉验证完成，汇总结果:")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(fold_results)
    
    # 计算平均性能
    avg_accuracy = results_df['accuracy'].mean()
    avg_mean_iou = results_df['mean_iou'].mean()
    
    print(f"平均准确率: {avg_accuracy:.4f}")
    print(f"平均IoU: {avg_mean_iou:.4f}")
    
    # 找出最佳模型（基于验证集平均IoU）
    best_fold_idx = results_df['mean_iou'].idxmax()
    best_fold = results_df.iloc[best_fold_idx]
    best_model_path = best_fold['best_model_path']
    
    print(f"最佳模型来自第 {best_fold['fold']} 折")
    print(f"最佳模型性能: 准确率={best_fold['accuracy']:.4f}, 平均IoU={best_fold['mean_iou']:.4f}")
    
    # 保存汇总结果
    summary_path = os.path.join(cv_output_dir, "cv_summary.csv")
    results_df.to_csv(summary_path, index=False)
    
    # 绘制所有折的性能比较图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(fold_results))
    width = 0.35
    plt.bar(x - width/2, results_df['accuracy'], width, label='准确率')
    plt.bar(x + width/2, results_df['mean_iou'], width, label='平均IoU')
    plt.axhline(y=avg_accuracy, color='r', linestyle='--', label=f'平均准确率: {avg_accuracy:.4f}')
    plt.axhline(y=avg_mean_iou, color='b', linestyle='--', label=f'平均IoU: {avg_mean_iou:.4f}')
    plt.xlabel('交叉验证折')
    plt.ylabel('性能指标')
    plt.title('交叉验证性能比较')
    plt.xticks(x, [f'折 {i+1}' for i in range(len(fold_results))])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cv_output_dir, 'cv_performance_comparison.png'))
    plt.close()
    
    # 对所有折计算每个类别的平均IoU
    avg_iou_per_class = np.zeros(num_classes)
    for metrics in all_val_metrics:
        avg_iou_per_class += metrics['iou_per_class']
    avg_iou_per_class /= len(all_val_metrics)
    
    # 绘制平均每类IoU
    class_names = [combined_class_names.get(i, f"类别{i}") for i in range(num_classes)]
    plot_iou_per_class(
        avg_iou_per_class,
        class_names,
        os.path.join(cv_output_dir, "average_iou_per_class.png")
    )
    
    # 返回最佳模型和平均指标
    return {
        'best_model_path': best_model_path,
        'avg_accuracy': avg_accuracy,
        'avg_mean_iou': avg_mean_iou,
        'avg_iou_per_class': avg_iou_per_class,
        'class_names': combined_class_names
    }

# 在主函数中添加对交叉验证的支持
if __name__ == "__main__":
    base_dir = r"D:\JimTemp\renet50\data"
    data_dir = os.path.join(base_dir, "masks")
    output_dir = os.path.join(base_dir, "output")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置文件路径
    config_file = os.path.join(base_dir, "train_configs.json")

    # 添加命令行参数支持交叉验证
    parser = argparse.ArgumentParser(description='DeepLabV3模型训练')
    parser.add_argument('--resume', action='store_true', help='从上次训练的模型继续训练')
    parser.add_argument('--model_path', type=str, default=None, help='要继续训练的模型路径')
    parser.add_argument('--cv', action='store_true', help='使用交叉验证')
    parser.add_argument('--folds', type=int, default=5, help='交叉验证折数')
    args = parser.parse_args()
    
    # 从配置文件加载参数
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            all_configs = config_data.get("configs", [{}])
            cv_config = all_configs[0] if all_configs else {}
    else:
        print("警告: 找不到配置文件，使用默认参数")
        cv_config = {}
    
    # 使用配置文件中的参数，如果没有则使用默认值
    num_epochs = cv_config.get("num_epochs", 30)
    batch_size = cv_config.get("batch_size", 4)
    lr = cv_config.get("lr", 0.0003)
    weight_decay = cv_config.get("weight_decay", 0.01)
    patience = cv_config.get("patience", 10)
    
    # 确认使用的参数
    print(f"使用的训练参数:num_epochs: {num_epochs}, batch_size: {batch_size}, lr: {lr}, weight_decay: {weight_decay}, patience: {patience}")

    cv_output_dir = os.path.join(output_dir, f"cv_{args.folds}fold_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(cv_output_dir, exist_ok=True)
    
    # 执行交叉验证 - 明确传递所有参数
    cross_validation(
        data_dir=data_dir,
        output_dir=cv_output_dir,
        num_folds=args.folds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        importance_factors=cv_config.get("importance_factors", None) if cv_config.get("use_importance_factors", False) else None
    )