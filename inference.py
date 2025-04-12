import torch
import cv2
from torchvision import transforms
import numpy as np
from torchvision import models
import os
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import json
import uuid
from werkzeug.utils import secure_filename
import io
import base64
from PIL import Image, ImageDraw, ImageFont
from flask_cors import CORS
import math

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用跨域资源共享
base_dir = r"D:\JimTemp\renet50\data"
# model_path = r"D:\seg\demo\best_model_fold3_28_135931_val_loss_1.0513.pth"
model_path = r"D:\JimTemp\renet50\data\11-40\3-28\best_model_28_114200_val_loss_1.2457.pth"

# 设置文件夹为绝对路径
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(base_dir, 'results')

# 确保上传和结果文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# 重要: 输出实际使用的路径，用于调试
print(f"上传文件夹: {app.config['UPLOAD_FOLDER']}")
print(f"结果文件夹: {app.config['RESULTS_FOLDER']}")


output_dir = os.path.join(base_dir, "./results")

# 修改标签和颜色映射的加载方式，直接指定文件路径
def load_label_and_color_maps(sample_folder=None):
    """
    从多个可能的来源加载标签和颜色映射:
    1. 指定的固定路径
    2. 样本文件夹内的label_color_map.json文件
    3. 默认的硬编码映射
    
    Args:
        sample_folder: 样本文件夹路径(可选)
    
    Returns:
        label_map: 标签ID到名称的映射
        color_map: 标签ID到颜色的映射
    """
    label_map = {}
    color_map = {}
    
    # 首先尝试从指定的固定路径加载
    fixed_json_path = os.path.join(base_dir, "label_color_map.json")
    if os.path.exists(fixed_json_path):
        try:
            print(f"尝试从固定路径加载映射: {fixed_json_path}")
            with open(fixed_json_path, 'r') as f:
                json_data = json.load(f)
                
            if "labels" in json_data and "colors" in json_data:
                # 直接使用JSON中的映射
                for label_id_str, label_name in json_data["labels"].items():
                    label_id = int(label_id_str)
                    label_map[label_id] = label_name
                
                for label_id_str, color in json_data["colors"].items():
                    label_id = int(label_id_str)
                    color_map[label_id] = color
                
                print(f"从固定路径成功加载了 {len(label_map)} 个标签和 {len(color_map)} 个颜色映射")
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
                    print(f"从固定路径成功加载了简化格式的映射")
                    return label_map, color_map
        except Exception as e:
            print(f"从固定路径加载映射时出错: {e}")
    
    return label_map, color_map



# 加载标签和颜色映射
LABEL_MAP, COLOR_MAP_DICT = load_label_and_color_maps(os.path.dirname(model_path))

# 确定类别数
num_classes = max(max(COLOR_MAP_DICT.keys()) + 1, 10)  # 使用最大类别ID或至少10个类别
print(f"加载模型: {model_path}, 类别数: {num_classes}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建模型并加载权重 - 使用ResNet101而不是ResNet50
model = models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))

try:
    # 加载模型权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("模型权重成功加载")
except Exception as e:
    print(f"加载模型权重时出错: {e}")
    exit(1)

model.to(device)
model.eval()  # 设置模型为评估模式

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 修改create_legend_with_chinese函数，使用1024x1024分辨率
def create_legend_with_chinese(color_map_dict, label_map, width=2048, height=512):
    """创建固定尺寸的中文图例，默认为横幅样式
    
    Args:
        color_map_dict: 颜色映射
        label_map: 标签映射
        width: 图例宽度，默认2048适合放在两张1024宽的图像下方
        height: 图例高度，默认512作为下方的图例
    
    Returns:
        固定尺寸的图例图像
    """
    # 创建白色背景图像
    legend = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(legend)
    
    # 尝试加载中文字体，更大的字体尺寸以适应更高分辨率
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
                font = ImageFont.truetype(path, 32)  # 增大字体尺寸以适应1024分辨率
                print(f"成功加载字体: {path}")
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
            print("警告：未找到中文字体，使用默认字体")
    except Exception as e:
        font = ImageFont.load_default()
        print(f"加载字体出错: {e}")
    
    # 计算每个图例项的高度
    total_items = len(label_map) + len([k for k in color_map_dict.keys() if k not in label_map])
    
    # 调整为横幅布局 - 计算每行能容纳多少项目和每个项目宽度
    items_per_row = 4  # 每行显示4个项目
    item_width = width // items_per_row
    item_height = min(70, (height - 60) // (math.ceil(total_items / items_per_row)))
    
    # 添加标题
    draw.text((20, 10), "类别图例", fill=(0,0,0), font=font)
    
    # 绘制图例项 - 按行排列
    current_x = 20
    current_y = 60  # 从标题下方开始
    item_count = 0
    
    # 首先绘制有标签的类别
    for idx, (class_idx, class_name) in enumerate(sorted(label_map.items())):
        # 计算当前项的位置
        row = item_count // items_per_row
        col = item_count % items_per_row
        
        pos_x = 20 + col * item_width
        pos_y = 60 + row * item_height
        
        if pos_y + item_height > height - 10:
            break  # 防止超出图例边界
            
        # 绘制颜色方块
        if class_idx in color_map_dict:
            color = tuple(color_map_dict[class_idx])
            draw.rectangle([(pos_x, pos_y), (pos_x + 50, pos_y + item_height-10)], 
                          fill=color, outline=(0,0,0))
            
            # 添加类别文本
            label_text = f"{class_name} (ID:{class_idx})"
            draw.text((pos_x + 60, pos_y + 5), label_text, fill=(0,0,0), font=font)
        
        item_count += 1
    
    # 绘制未在label_map中但在color_map中的类别
    for class_idx, color in sorted(color_map_dict.items()):
        if class_idx not in label_map:
            # 计算当前项的位置
            row = item_count // items_per_row
            col = item_count % items_per_row
            
            pos_x = 20 + col * item_width
            pos_y = 60 + row * item_height
            
            if pos_y + item_height > height - 10:
                break  # 防止超出图例边界
                
            # 绘制颜色方块
            draw.rectangle([(pos_x, pos_y), (pos_x + 50, pos_y + item_height-10)], 
                          fill=tuple(color), outline=(0,0,0))
            
            # 添加类别文本
            label_text = f"未知类别 (ID:{class_idx})"
            draw.text((pos_x + 60, pos_y + 5), label_text, fill=(0,0,0), font=font)
            
            item_count += 1
    
    # 转换为numpy数组以与OpenCV兼容
    return np.array(legend)

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

# 修改process_image函数使用1024x1024的分辨率
def process_image(img, timestamp=None):
    """处理单个图像并返回分割结果"""
    if timestamp is None:
        timestamp = str(uuid.uuid4())
    
    sample_folder = os.path.join(base_dir, "label_color_map.json")
    label_map, color_map_dict = load_label_and_color_maps(sample_folder)
    
    # 将颜色映射字典转换为NumPy数组
    max_label = max(max(color_map_dict.keys()) + 1, model.classifier[4].out_channels)
    COLOR_MAP = np.zeros((max_label, 3), dtype=np.uint8)
    
    # 为所有没有指定颜色的类别生成随机颜色，但对背景类0使用固定颜色
    for i in range(max_label):
        if i == 0:  # 背景类固定为黑色
            COLOR_MAP[i] = np.array([0, 0, 0])
        elif i in color_map_dict:
            COLOR_MAP[i] = color_map_dict[i]
        else:
            # 为未指定颜色的类别生成随机颜色，但使用固定的随机种子
            np.random.seed(i)  # 使用类别ID作为随机种子
            COLOR_MAP[i] = np.random.randint(0, 256, 3)
    
    # 恢复随机种子
    np.random.seed(None)
    
    # 预处理并将所有图像调整为1024x1024
    img_resized = cv2.resize(img, (1024, 1024))
    
    # 为模型预测创建512x512的图像副本(保持模型输入不变)
    img_for_model = cv2.resize(img_resized, (512, 512))
    img_tensor = transform(img_for_model).unsqueeze(0).to(device)
    
    # 确保模型在评估模式
    model.eval()
    
    with torch.no_grad():
        output = model(img_tensor)['out']
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # 应用平滑处理
    pred = smooth_prediction(pred, num_classes=model.classifier[4].out_channels)
    
    # 将预测结果放大到1024x1024
    pred = cv2.resize(pred.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST)
    
    # 创建彩色掩码 - 确保索引不超出COLOR_MAP的范围
    pred_safe = np.clip(pred, 0, max_label-1)
    
    # 使用动态加载的颜色映射为掩码着色
    colored_pred = COLOR_MAP[pred_safe].astype(np.uint8)
    
    # 创建叠加图 - 使用img_resized，确保尺寸匹配
    alpha = 0.6
    overlay = cv2.addWeighted(img_resized, 1-alpha, colored_pred, alpha, 0)
    
    # 确保所有输出图像都是1024x1024
    img_display = img_resized  # 直接使用已经调整过的图像
    
    # 为当前样本创建输出目录
    sample_output_dir = os.path.join(app.config['RESULTS_FOLDER'], timestamp)
    if not os.path.exists(sample_output_dir):
        os.makedirs(sample_output_dir)
    
    # 保存结果
    cv2.imwrite(os.path.join(sample_output_dir, 'original.png'), cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(sample_output_dir, 'pred_mask.png'), cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(sample_output_dir, 'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # 创建固定尺寸的图例 (2048x512)，使其宽度与上方两张图的总宽度匹配
    legend = create_legend_with_chinese(color_map_dict, label_map, width=2048, height=512)
    
    # 创建垂直组合的图像：上面是原图和掩码并排，下面是图例
    # 总尺寸: 1024+512 高 x 2048 宽 = 1536 x 2048
    combined_img = np.ones((1024+512, 2048, 3), dtype=np.uint8) * 255
    
    # 添加原图 (左上角)
    combined_img[0:1024, 0:1024] = img_display
    
    # 添加预测掩码 (右上角)
    combined_img[0:1024, 1024:2048] = colored_pred
    
    # 添加图例 (下方，跨越整个宽度)
    combined_img[1024:1024+512, 0:2048] = legend
    
    # 保存组合图像
    cv2.imwrite(os.path.join(sample_output_dir, 'combined.png'), 
               cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
    
    return {
        'timestamp': timestamp,
        'original': 'original.png',
        'pred_mask': 'pred_mask.png',
        'overlay': 'overlay.png',
        'combined': 'combined.png'
    }

# 创建Flask路由
@app.route('/')
def index():
    """主页 - 显示图像上传表单"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理上传的图像文件"""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # 创建唯一的文件名
        timestamp = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 处理图像
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调用处理函数
        results = process_image(img, timestamp)
        
        # 重定向到结果页面
        return redirect(url_for('results', timestamp=timestamp))

@app.route('/results/<timestamp>')
def results(timestamp):
    """显示处理结果"""
    results_dir = os.path.join(app.config['RESULTS_FOLDER'], timestamp)
    if not os.path.exists(results_dir):
        return "结果不存在", 404
    
    # 获取所有结果图像
    image_files = {
        'original': 'original.png',
        'pred_mask': 'pred_mask.png',
        'overlay': 'overlay.png',
        'combined': 'combined.png'
    }
    
    return render_template('result.html', 
                           timestamp=timestamp, 
                           images=image_files)

@app.route('/results/<timestamp>/<filename>')
def result_file(timestamp, filename):
    """提供结果文件"""
    # 使用os.path.abspath确保路径正确无论从哪里启动应用
    result_dir = os.path.abspath(os.path.join(app.config['RESULTS_FOLDER'], timestamp))
    return send_from_directory(result_dir, filename, as_attachment=False)

# 添加基于AJAX的处理方法，无需刷新页面就能看到结果
@app.route('/process_image', methods=['POST'])
def process_image_ajax():
    """使用AJAX处理上传的图像"""
    if 'file' not in request.files:
        return json.dumps({'error': '没有发现文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return json.dumps({'error': '未选择文件'}), 400
    
    # 读取图像
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 处理图像
    timestamp = str(uuid.uuid4())
    results = process_image(img, timestamp)
    
    # 准备结果URL
    result_urls = {}
    for key, filename in results.items():
        if key != 'timestamp':
            result_urls[key] = url_for('result_file', timestamp=timestamp, filename=filename)
    
    return json.dumps({
        'success': True,
        'timestamp': timestamp,
        'result_urls': result_urls
    })

@app.route('/test')
def test_paths():
    """测试文件系统和路径"""
    results = {
        'server_info': {
            'working_dir': os.getcwd(),
            'script_dir': os.path.dirname(os.path.abspath(__file__)),
        },
        'config': {
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'results_folder': app.config['RESULTS_FOLDER'],
        },
        'results_content': []
    }
    
    # 检查results文件夹
    if os.path.exists(app.config['RESULTS_FOLDER']):
        results['results_content'] = os.listdir(app.config['RESULTS_FOLDER'])
        
        # 检查第一个结果文件夹(如果存在)
        if results['results_content']:
            first_result = os.path.join(app.config['RESULTS_FOLDER'], results['results_content'][0])
            if os.path.isdir(first_result):
                results['first_result_files'] = os.listdir(first_result)
    
    return json.dumps(results, indent=2)

# 主程序运行
if __name__ == "__main__":

    # 运行Flask应用
    app.run(debug=True, host='0.0.0.0', port=5003)