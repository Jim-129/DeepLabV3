import os
import json
import numpy as np
import cv2
from pathlib import Path
import glob

# 设置输入输出路径
IMAGES_DIR = r"D:\JimTemp\renet50\data\images"
JSON_DIR = r"D:\JimTemp\renet50\data\json"
MASKS_DIR = r"D:\JimTemp\renet50\data\masks"
LABEL_INFO_FILE = r"D:\JimTemp\renet50\data\label_color_map.json"  # 创建标签信息文件

# 创建输出目录
os.makedirs(MASKS_DIR, exist_ok=True)

# 初始化类别映射和颜色映射
class_mapping = {}
color_mapping = {}
class_counter = 1  # 0保留给背景

# 获取所有JSON文件
json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))

# 第一次遍历，收集所有类别
print("第一步：收集所有标签类别...")
for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for shape in data['shapes']:
        label = shape['label']
        if label not in class_mapping:
            class_mapping[label] = class_counter
            # 为每个类别生成一个随机颜色
            color_mapping[class_counter] = [
                np.random.randint(0, 256), 
                np.random.randint(0, 256), 
                np.random.randint(0, 256)
            ]
            class_counter += 1

# 保存标签和颜色映射
label_color_map = {
    "labels": {str(v): k for k, v in class_mapping.items()},  # ID -> 名称
    "colors": {str(k): v for k, v in color_mapping.items()}   # ID -> 颜色
}

with open(LABEL_INFO_FILE, 'w', encoding='utf-8') as f:
    json.dump(label_color_map, f, indent=4)

print(f"标签和颜色映射已保存到 {LABEL_INFO_FILE}，共 {len(class_mapping)} 个类别")
print("类别映射:")
for label, id in class_mapping.items():
    print(f"  - {label}: {id}")

# 第二次遍历，生成多类别掩码
print("\n第二步：生成多类别掩码...")
for json_file in json_files:
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取图像信息
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    
    # 创建空白mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # 处理每个标注区域
    for shape in data['shapes']:
        # 获取标签和对应的类别ID
        label = shape['label']
        class_id = class_mapping[label]
        
        # 获取多边形点
        points = shape['points']
        points = np.array(points, dtype=np.int32)
        
        # 绘制多边形填充区域，使用类别ID作为填充值
        cv2.fillPoly(mask, [points], class_id)
    
    # 保存mask文件
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    mask_path = os.path.join(MASKS_DIR, base_name + '_mask.png')
    cv2.imwrite(mask_path, mask)
    
    print(f"已处理: {json_file} -> {mask_path}")

print("所有多类别掩码生成完成！")

# 可视化示例 - 创建一个彩色掩码图像用于可视化检查
print("\n第三步：生成可视化示例...")
for mask_file in glob.glob(os.path.join(MASKS_DIR, "*_mask.png")):
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    
    # 创建彩色掩码
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # 为每个类别上色
    for class_id, color in color_mapping.items():
        colored_mask[mask == class_id] = color
    
    # 保存彩色掩码
    vis_file = mask_file.replace('_mask.png', '_colored_mask.png')
    cv2.imwrite(vis_file, colored_mask)
    print(f"生成可视化掩码: {vis_file}")

print("处理完成！")