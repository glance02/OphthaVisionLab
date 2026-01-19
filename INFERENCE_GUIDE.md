# 二分类模型使用指南

## 📁 训练好的模型文件

| 数据集 | 模型路径 | 最佳F1分数 |
|--------|----------|------------|
| dataset260106 | `./myProcessResults/single_cls_260106_Binary_finetune/checkpoint_teacher_linear.pth` | 0.8047 |
| IDRiD | `./IDRiDResults/IDRiD_Binary_finetune/checkpoint_teacher_linear.pth` | 0.7984 |

## 🚀 使用方法

### 方法1：使用独立推理脚本（推荐）

#### 单张图像预测
```bash
python3 inference_binary.py \
    --checkpoint ./myProcessResults/single_cls_260106_Binary_finetune/checkpoint_teacher_linear.pth \
    --image /path/to/your/image.jpg \
    --finetune
```

**输出示例**：
```
==================================================
预测图像: /path/to/your/image.jpg
==================================================
Logit: 1.2345
Probability: 0.7742
Predicted Class: 1 (Positive)
```

#### 批量图像预测
```bash
python3 inference_binary.py \
    --checkpoint ./myProcessResults/single_cls_260106_Binary_finetune/checkpoint_teacher_linear.pth \
    --image_dir /path/to/image/folder \
    --output_file predictions.txt \
    --finetune
```

**输出示例**：
```
找到 100 张图像
正在预测...

image_001.jpg                  | Prob: 0.2341 | Class: 0
image_002.jpg                  | Prob: 0.8756 | Class: 1
...

结果已保存到: predictions.txt
```

### 方法2：使用训练脚本的测试模式

```bash
bash test_model.sh
```

## 📝 参数说明

| 参数 | 说明 | 必需 |
|------|------|------|
| `--checkpoint` | 训练好的checkpoint路径 | ✓ |
| `--pretrained_weights` | 预训练权重路径 | - |
| `--image` | 单张图像路径 | - |
| `--image_dir` | 图像文件夹路径 | - |
| `--output_file` | 结果保存路径 | - |
| `--finetune` | 是否使用fine-tuning模型 | - |

## 🔍 结果解读

### 输出指标

1. **Logit**: 原始模型输出（未经过sigmoid）
   - 正值表示倾向于类别1（阳性）
   - 负值表示倾向于类别0（健康）

2. **Probability**: 经过sigmoid转换的概率（0-1之间）
   - 接近1：高概率为阳性
   - 接近0：高概率为健康
   - 默认阈值：0.5

3. **Prediction**: 最终预测类别
   - 0: 健康（Healthy）
   - 1: 阳性（DR）

### 调整预测阈值

如果需要调整阳性判定的阈值，可以修改 `inference_binary.py` 中的第107行：
```python
# 原始：pred_class = 1 if prob > 0.5 else 0
# 更高精度（减少假阳性）：
pred_class = 1 if prob > 0.7 else 0
# 更高召回率（减少假阴性）：
pred_class = 1 if prob > 0.3 else 0
```

## 💡 使用建议

1. **模型选择**：
   - 如果测试数据与dataset260106类似，使用dataset260106模型
   - 如果测试数据质量较高，建议使用IDRiD模型

2. **输入要求**：
   - 支持格式：JPG, PNG
   - 图像会被自动resize到224×224
   - RGB三通道图像

3. **性能优化**：
   - 批量预测时，GPU会持续使用，效率更高
   - 首次加载模型需要几秒钟，后续预测很快

## 🐛 常见问题

**Q: 为什么预测概率总是接近0或1？**
A: 这是模型经过训练后的正常表现，说明模型对预测比较有信心。

**Q: 如何处理多张图像？**
A: 使用 `--image_dir` 参数指定文件夹，会自动处理所有jpg和png图像。

**Q: 可以在CPU上运行吗？**
A: 可以，但需要修改代码将 `.cuda()` 相关调用去掉，速度会较慢。
