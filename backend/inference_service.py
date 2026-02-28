"""
通用推理服务
支持二分类、多分类、分割任务
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
import io
import cv2
import sys

from model_factory import ModelFactory


class InferenceService:
    """通用推理服务 - 支持多种任务"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.factory = ModelFactory(self.device)
        self.models = {}  # 缓存已加载的模型

        print(f"推理服务初始化完成，使用设备: {self.device}")

    def _get_or_create_model(self, task_type, task_id, **kwargs):
        """获取或创建模型（带缓存）"""
        cache_key = f"{task_type}_{task_id}"

        if cache_key not in self.models:
            print(f"加载模型: {task_type} - {task_id}")

            if task_type == 'segmentation':
                model = self.factory.create_segmentation_model(
                    checkpoint_path=kwargs.get('checkpoint'),
                    pretrained_weights=kwargs.get('pretrained_weights', True),
                    input_size=kwargs.get('input_size', 512),
                    num_classes=kwargs.get('num_classes', 1)
                )
            elif task_type == 'binary':
                model = self.factory.create_binary_classifier(
                    checkpoint_path=kwargs.get('checkpoint'),
                    pretrained_weights=kwargs.get('pretrained_weights', True),
                    input_size=kwargs.get('input_size', 224),
                    n_last_blocks=kwargs.get('n_last_blocks', 4),
                    avgpool=kwargs.get('avgpool', 0),
                    finetune=kwargs.get('finetune', True)
                )
            elif task_type == 'multiclass':
                model = self.factory.create_multiclass_classifier(
                    checkpoint_path=kwargs.get('checkpoint'),
                    num_labels=kwargs.get('num_labels'),
                    pretrained_weights=kwargs.get('pretrained_weights', True),
                    input_size=kwargs.get('input_size', 224),
                    n_last_blocks=kwargs.get('n_last_blocks', 4),
                    avgpool=kwargs.get('avgpool', 0),
                    finetune=kwargs.get('finetune', True)
                )
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")

            self.models[cache_key] = model

        return self.models[cache_key]

    def preprocess_image(self, image_bytes: bytes, input_size=224, modality='Fundus'):
        """预处理图像"""
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)

        # CLAHE 增强（如果图像过暗）
        if img_array.mean() / 255.0 < 0.15:
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)

            lab_clahe = cv2.merge([l_channel, a, b])
            img_array = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            img = Image.fromarray(img_array)

        # 获取统计量
        mean, std = self._get_stats(modality)

        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        return transform(img).unsqueeze(0)

    def _get_stats(self, modality):
        """获取图像统计量"""
        # 从 utils 导入
        import utils
        return utils.get_stats(modality)

    def predict_segmentation(self, image_bytes: bytes, checkpoint_path,
                             input_size=512, threshold=0.5):
        """分割预测"""
        model = self._get_or_create_model(
            task_type='segmentation',
            task_id=Path(checkpoint_path).stem,
            checkpoint=checkpoint_path,
            input_size=input_size
        )

        # 预处理
        input_tensor = self.preprocess_image(image_bytes, input_size, 'Fundus')

        # 推理
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = model(input_tensor)
            prob = torch.sigmoid(output)
            prob_np = prob.squeeze().cpu().numpy()

        # 生成掩码图像
        mask = (prob_np > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask)
        mask_bytes = io.BytesIO()
        mask_img.save(mask_bytes, format='PNG')

        return {
            'mask': mask_bytes.getvalue(),
            'probability_map': prob_np,
            'shape': (input_size, input_size),
            'threshold': threshold
        }

    def predict_binary(self, image_bytes: bytes, checkpoint_path,
                       input_size=224, n_last_blocks=4, avgpool=0):
        """二分类预测"""
        model = self._get_or_create_model(
            task_type='binary',
            task_id=Path(checkpoint_path).stem,
            checkpoint=checkpoint_path,
            input_size=input_size,
            n_last_blocks=n_last_blocks,
            avgpool=avgpool
        )

        # 预处理
        input_tensor = self.preprocess_image(image_bytes, input_size, 'Fundus')

        # 推理
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            logit = model(input_tensor).squeeze()
            prob = torch.sigmoid(logit).item()
            pred_class = 1 if prob > 0.5 else 0

        return {
            'logit': float(logit),
            'probability': prob,
            'predicted_class': pred_class,
            'confidence': prob if pred_class == 1 else 1 - prob
        }

    def predict_multiclass(self, image_bytes: bytes, checkpoint_path, num_labels,
                           input_size=224, n_last_blocks=4, avgpool=0):
        """多分类预测"""
        model = self._get_or_create_model(
            task_type='multiclass',
            task_id=Path(checkpoint_path).stem,
            checkpoint=checkpoint_path,
            num_labels=num_labels,
            input_size=input_size,
            n_last_blocks=n_last_blocks,
            avgpool=avgpool
        )

        # 预处理
        input_tensor = self.preprocess_image(image_bytes, input_size, 'Fundus')

        # 推理
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze()
            probs_np = probs.cpu().numpy()

            pred_class = int(probs_np.argmax())
            confidence = float(probs_np[pred_class])

        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': probs_np.tolist(),
            'num_classes': num_labels
        }


# 全局服务实例
inference_service = InferenceService()
