"""
使用 backend 模型测试 IDRiD 数据集
"""
import sys
sys.path.insert(0, './backend')

import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
import utils

# 配置
CHECKPOINT_PATH = './backend/checkpoints/single_cls/checkpoint_teacher_linear.pth'
PRETRAINED_WEIGHTS = './backend/pretrain_weights/VFM_Fundus_weights.pth'
DATA_PATH = './dataset/SingleModalCls/FundusClassification/IDRiD'
INPUT_SIZE = 224
N_LAST_BLOCKS = 4
AVGPOOL = 0

# 加载模型
from model_factory import ModelFactory, ClsModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模型
factory = ModelFactory(device)
encoder = factory.create_encoder('vit_base', 16, INPUT_SIZE)

# 加载预训练权重
utils.load_pretrained_weights(
    encoder,
    PRETRAINED_WEIGHTS,
    'teacher',
    'vit_base',
    16
)

# 创建分类头
from models.head import ClsHead
embed_dim = getattr(encoder, "embed_dim", 768)
head = ClsHead(embed_dim=embed_dim * N_LAST_BLOCKS, num_classes=1, layers=3)

# 加载 checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state_dict[new_key] = v
    head.load_state_dict(new_state_dict)
    print(f"✓ 分类头权重加载成功")

# 组合模型
model = ClsModel(encoder, head, N_LAST_BLOCKS, AVGPOOL).to(device).eval()

# 获取数据统计量
mean, std = utils.get_stats('Fundus')

# 加载测试数据
test_labels_file = Path(DATA_PATH) / 'test_labels.txt'
test_images_dir = Path(DATA_PATH) / 'test'

images = []
labels = []
predictions = []
probabilities = []

print(f"\n加载测试数据 from {test_labels_file}")

with open(test_labels_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_path, label = line.split(';')
        img_path = img_path.replace('test/', '')  # 去掉 test/ 前缀
        full_path = test_images_dir / img_path

        if not full_path.exists():
            print(f"警告: 文件不存在 {full_path}")
            continue

        # 读取并预处理图像
        img = Image.open(full_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            logit = model(img_tensor).squeeze()
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob > 0.5 else 0

        images.append(str(full_path))
        labels.append(int(label))
        predictions.append(pred)
        probabilities.append(prob)

        if len(predictions) % 50 == 0:
            print(f"已处理 {len(predictions)} 张图像")

print(f"\n总共处理了 {len(predictions)} 张图像")

# 计算指标
labels = np.array(labels)
predictions = np.array(predictions)
probabilities = np.array(probabilities)

acc = accuracy_score(labels, predictions) * 100

# 处理只有一个类别的情况
unique_labels = np.unique(labels)
if len(unique_labels) == 2:
    auc = roc_auc_score(labels, probabilities) * 100
else:
    auc = 0.0
    print("警告: 只有一类标签，无法计算 AUC")

f1 = f1_score(labels, predictions) * 100
precision = precision_score(labels, predictions, zero_division=0) * 100
recall = recall_score(labels, predictions, zero_division=0) * 100
ap = average_precision_score(labels, probabilities) * 100

print("\n" + "="*50)
print("测试结果:")
print("="*50)
print(f"Acc@1: {acc:.2f}%")
print(f"AUC:   {auc:.2f}%")
print(f"F1:    {f1:.2f}%")
print(f"Pre:   {precision:.2f}%")
print(f"Recall:{recall:.2f}%")
print(f"AP:    {ap:.2f}%")
print("="*50)

# 保存结果
import json
results = {
    'acc': float(acc),
    'auc': float(auc),
    'f1': float(f1),
    'precision': float(precision),
    'recall': float(recall),
    'ap': float(ap),
    'num_samples': len(predictions)
}

output_path = './backend/checkpoints/single_cls/idrid_test_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n结果已保存到 {output_path}")
