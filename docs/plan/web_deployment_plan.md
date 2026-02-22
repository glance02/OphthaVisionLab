# VisionFM 分割模型 Web 应用部署计划（本地演示版）

## 项目概述

将 VisionFM 眼科图像分割模型封装为本地 Web 应用，支持用户上传眼底图像并获得实时的血管分割结果。

---

## 系统架构（简化版）

```
┌────────────────────────────────────────────────────┐
│              浏览器 (localhost)                      │
│  ┌──────────────────────────────────────────────┐  │
│  │         前端 (Vue 3 + Element Plus)            │  │
│  └──────────────────────────────────────────────┘  │
│                      │ HTTP                          │
│                      ▼                               │
│  ┌──────────────────────────────────────────────┐  │
│  │          后端 (FastAPI)                        │  │
│  │  - 同步推理  - 内存缓存                        │  │
│  └──────────────────────────────────────────────┘  │
│                      │                               │
│                      ▼                               │
│  ┌──────────────────────────────────────────────┐  │
│  │        VisionFM 模型 (GPU/CPU)                 │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

---

## 一、前端设计

### 1.1 技术选型

| 技术 | 版本 | 说明 |
|------|------|------|
| 框架 | Vue 3 | Composition API |
| 组件库 | Element Plus | UI 组件 |
| HTTP | Axios | 请求封装 |
| 开发 | Vite | 构建工具 |

### 1.2 安装依赖

```bash
npm create vue@latest visionfm-frontend
cd visionfm-frontend
npm install element-plus axios
```

### 1.3 项目结构

```
visionfm-frontend/
├── src/
│   ├── components/
│   │   ├── ImageUpload.vue      # 图片上传组件
│   │   ├── ResultDisplay.vue    # 结果展示组件
│   │   └── ControlPanel.vue     # 控制面板
│   ├── App.vue
│   └── main.js
├── public/
└── package.json
```

### 1.4 页面布局

```
┌────────────────────────────────────────────────────────┐
│     VisionFM 眼科图像分割演示                    [Logo] │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐        ┌──────────────────┐     │
│  │   图片上传区      │   →    │   分割结果展示    │     │
│  │                  │        │                  │     │
│  │  [点击或拖拽上传] │        │  [原图 + 掩码]   │     │
│  │                  │        │                  │     │
│  │  支持 JPG/PNG    │        │  透明度: [━━○──] │     │
│  │  最大 10MB       │        │  [下载结果]      │     │
│  └──────────────────┘        └──────────────────┘     │
│                                                         │
│  [处理中...] ████████████░░░░ 75%                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.5 核心组件代码

#### 1.5.1 图片上传组件 (`ImageUpload.vue`)

```vue
<template>
  <el-upload
    class="upload-area"
    drag
    :auto-upload="false"
    :show-file-list="false"
    accept=".jpg,.jpeg,.png"
    :on-change="handleFileChange"
    :limit="1"
  >
    <el-icon class="upload-icon"><UploadFilled /></el-icon>
    <div class="upload-text">拖拽图片到此处或点击上传</div>
    <div class="upload-hint">支持 JPG/PNG 格式，最大 10MB</div>
  </el-upload>
</template>

<script setup>
import { UploadFilled } from '@element-plus/icons-vue';

const emit = defineEmits(['upload']);

const handleFileChange = (file) => {
  // 验证大小
  if (file.raw.size > 10 * 1024 * 1024) {
    ElMessage.error('图片大小不能超过 10MB');
    return;
  }
  emit('upload', file.raw);
};
</script>

<style scoped>
.upload-area {
  padding: 40px;
  border: 2px dashed #d9d9d9;
  border-radius: 8px;
  text-align: center;
}
.upload-icon {
  font-size: 48px;
  color: #409eff;
}
</style>
```

#### 1.5.2 结果展示组件 (`ResultDisplay.vue`)

```vue
<template>
  <div class="result-container">
    <div class="image-comparison">
      <!-- 原图 -->
      <div class="image-box">
        <h4>原图</h4>
        <img :src="originalImage" alt="原图" />
      </div>
      <!-- 叠加结果 -->
      <div class="image-box">
        <h4>分割结果</h4>
        <div class="overlay-wrapper">
          <img :src="originalImage" alt="原图" class="base-image" />
          <img :src="maskImage" alt="掩码" class="mask-image"
               :style="{ opacity: opacity }" />
        </div>
      </div>
    </div>

    <div class="controls">
      <span>透明度: {{ (opacity * 100).toFixed(0) }}%</span>
      <el-slider v-model="opacity" :min="0" :max="1" :step="0.01" />
      <el-button type="primary" @click="downloadResult">下载结果</el-button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const props = defineProps(['originalImage', 'maskImage']);
const opacity = ref(0.6);

const downloadResult = () => {
  // 叠加下载逻辑
};
</script>

<style scoped>
.image-comparison {
  display: flex;
  gap: 20px;
}
.image-box {
  flex: 1;
}
.overlay-wrapper {
  position: relative;
}
.mask-image {
  position: absolute;
  top: 0;
  left: 0;
}
</style>
```

#### 1.5.3 主应用 (`App.vue`)

```vue
<template>
  <div class="app">
    <el-container>
      <el-header>
        <h1>VisionFM 眼科图像分割演示</h1>
      </el-header>

      <el-main>
        <div v-if="!processing && !result" class="upload-section">
          <ImageUpload @upload="handleUpload" />
        </div>

        <div v-if="processing" class="processing">
          <el-progress :percentage="progress" :status="progressStatus" />
          <p>{{ statusMessage }}</p>
        </div>

        <div v-if="result" class="result-section">
          <ResultDisplay
            :original-image="result.originalImage"
            :mask-image="result.maskImage"
          />
          <el-button @click="reset">重新上传</el-button>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';
import ImageUpload from './components/ImageUpload.vue';
import ResultDisplay from './components/ResultDisplay.vue';

const API_BASE = 'http://localhost:8000';

const processing = ref(false);
const progress = ref(0);
const progressStatus = ref('');
const statusMessage = ref('');
const result = ref(null);

const handleUpload = async (file) => {
  processing.value = true;
  progress.value = 0;
  statusMessage.value = '上传中...';

  const formData = new FormData();
  formData.append('file', file);

  try {
    // 同步请求，等待处理完成
    const response = await axios.post(`${API_BASE}/api/segment`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        progress.value = Math.round((e.loaded / e.total) * 30);
      }
    });

    progress.value = 100;
    result.value = response.data;
    ElMessage.success('分割完成！');
  } catch (error) {
    ElMessage.error('处理失败: ' + error.message);
  } finally {
    processing.value = false;
  }
};

const reset = () => {
  result.value = null;
  progress.value = 0;
};
</script>
```

---

## 二、后端设计

### 2.1 技术选型

| 技术 | 版本 | 说明 |
|------|------|------|
| 框架 | FastAPI 0.104+ | Web 框架 |
| 深度学习 | PyTorch | 模型推理 |
| 图像处理 | Pillow, OpenCV | 图像处理 |
| CORS | fastapi.middleware.cors | 跨域支持 |

### 2.2 安装依赖

```bash
pip install fastapi uvicorn torch torchvision pillow opencv-python python-multipart
```

### 2.3 项目结构

```
visionfm-backend/
├── main.py                    # FastAPI 主入口
├── model_service.py           # 模型服务
├── models/                    # 模型定义（从原项目复制）
│   ├── vision_transformer.py
│   └── unetr_head.py
├── utils.py                   # 工具函数（从原项目复制）
├── pretrain_weights/          # 预训练权重
│   └── VFM_Fundus_weights.pth
└── checkpoints/               # 训练权重
    └── checkpoint_108_linear.pth
```

### 2.4 核心代码

#### 2.4.1 模型服务 (`model_service.py`)

```python
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import sys
from pathlib import Path

# 添加项目路径
proj_root = Path(__file__).parent
sys.path.insert(0, str(proj_root))

import models
import utils
from models.unetr_head import Unetr_Head


class SegModel(nn.Module):
    """分割模型封装"""
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        n = len(self.encoder.blocks)
        inter = self.encoder.get_intermediate_layers(x, n)

        # 选择层索引
        if n == 12:
            selected = [3, 5, 7, 11]
        else:
            selected = list(range(max(0, n - 4), n))

        feats = [inter[idx][:, 1:] for idx in selected]
        out = self.head(feats, x)
        return out


class SegmentationService:
    """单例模型服务"""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {self.device}")
            self._load_model()

    def _load_model(self):
        """加载模型"""
        # Encoder
        encoder = models.vit_base(
            img_size=[512],
            patch_size=16,
            num_classes=0
        )

        utils.load_pretrained_weights(
            encoder,
            'pretrain_weights/VFM_Fundus_weights.pth',
            'teacher',
            'vit_base',
            16
        )

        # Head
        head = Unetr_Head(embed_dim=768, num_classes=1, img_dim=512)
        checkpoint = torch.load('checkpoints/checkpoint_108_linear.pth', map_location='cpu')

        # 去掉 'module.' 前缀
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith('module.') else k
            if new_key.startswith('head.'):
                new_state_dict[new_key[5:]] = v

        head.load_state_dict(new_state_dict, strict=False)

        # 组合模型
        self._model = SegModel(encoder, head).to(self.device).eval()
        print("✓ 模型加载成功")

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """预处理图像"""
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)

        # CLAHE 增强（如果图像过暗）
        import cv2
        if img_array.mean() / 255.0 < 0.15:
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img_array = cv2.merge([l, a, b])
            img_array = cv2.cvtColor(img_array, cv2.COLOR_LAB2RGB)
            img = Image.fromarray(img_array)

        mean, std = utils.get_stats('Fundus')
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transform(img).unsqueeze(0)

    def predict(self, image_bytes: bytes) -> bytes:
        """推理并返回掩码图像 bytes"""
        # 预处理
        input_tensor = self.preprocess(image_bytes)

        # 推理
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self._model(input_tensor)
            prob = torch.sigmoid(output)
            prob_np = prob.squeeze().cpu().numpy()

        # 生成掩码图像
        mask = (prob_np > 0.5).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask)
        mask_bytes = io.BytesIO()
        mask_img.save(mask_bytes, format='PNG')
        return mask_bytes.getvalue()


# 全局实例
seg_service = SegmentationService()
```

#### 2.4.2 主应用 (`main.py`)

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import base64
import io

from model_service import seg_service

app = FastAPI(title="VisionFM Segmentation API")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "VisionFM Segmentation API", "model": "loaded"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": str(seg_service.device)
    }


@app.post("/api/segment")
async def segment_image(file: UploadFile = File(...)):
    """图像分割接口"""
    # 验证文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="只支持图片文件")

    # 验证大小
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片大小不能超过 10MB")

    # 推理
    try:
        mask_bytes = seg_service.predict(content)

        # 转换为 base64
        original_b64 = base64.b64encode(content).decode('utf-8')
        mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')

        return {
            "success": True,
            "originalImage": f"data:image/png;base64,{original_b64}",
            "maskImage": f"data:image/png;base64,{mask_b64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2.5 运行后端

```bash
# 进入后端目录
cd visionfm-backend

# 启动服务
python main.py
# 或使用 uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

访问: http://localhost:8000/docs 查看 API 文档

---

## 三、文件准备

### 3.1 从原项目复制文件

```bash
# 复制模型定义
cp models/vision_transformer.py visionfm-backend/models/
cp models/unetr_head.py visionfm-backend/models/

# 复制工具函数
cp utils.py visionfm-backend/

# 复制权重文件
cp -r pretrain_weights/ visionfm-backend/
cp results/single_seg_debug/checkpoint_108_linear.pth visionfm-backend/checkpoints/
```

### 3.2 目录结构确认

```
visionfm-backend/
├── main.py
├── model_service.py
├── utils.py
├── models/
│   ├── __init__.py          # 需要创建
│   ├── vision_transformer.py
│   └── unetr_head.py
├── pretrain_weights/
│   └── VFM_Fundus_weights.pth
└── checkpoints/
    └── checkpoint_108_linear.pth
```

---

## 四、运行步骤

### 4.1 后端启动

```bash
cd visionfm-backend
python main.py
```

### 4.2 前端启动

```bash
cd visionfm-frontend
npm run dev
```

### 4.3 访问应用

打开浏览器访问: http://localhost:5173

---

## 五、快速启动脚本

### 5.1 Windows (`start.bat`)

```batch
@echo off
echo 启动 VisionFM 演示系统...

start "Backend" cmd /k "cd visionfm-backend && python main.py"
timeout /t 5 /nobreak > nul
start "Frontend" cmd /k "cd visionfm-frontend && npm run dev"

echo 服务已启动！
echo 后端: http://localhost:8000
echo 前端: http://localhost:5173
```

### 5.2 Linux/Mac (`start.sh`)

```bash
#!/bin/bash
cd visionfm-backend && python main.py &
BACKEND_PID=$!
sleep 5
cd visionfm-frontend && npm run dev &
FRONTEND_PID=$!

echo "后端 PID: $BACKEND_PID"
echo "前端 PID: $FRONTEND_PID"
echo "访问 http://localhost:5173"
```

---

## 六、开发时间线（简化版）

| 任务 | 时间 |
|------|------|
| 后端 API 开发 | 1 天 |
| 前端组件开发 | 2 天 |
| 联调测试 | 0.5 天 |
| UI 优化 | 0.5 天 |
| **总计** | **4 天** |

---

## 七、注意事项

1. **模型权重文件较大**，确保有足够磁盘空间
2. **GPU 可选**，没有 GPU 会自动使用 CPU（较慢）
3. **首次运行**会加载模型，需要等待几秒
4. **图片格式**仅支持 JPG/PNG
5. **图片大小**建议不超过 5MB，处理更快

---

*文档生成时间: 2026-02-11*
*本地演示版 - 简化架构*
