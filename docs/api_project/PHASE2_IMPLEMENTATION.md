# VisionFM AI 分析系统 - 第二阶段实现文档

> 完成日期：2026-02-28  
> 阶段：第二阶段 - 后端开发  
> 状态：✅ 完成

---

## 一、阶段概述

第二阶段的目标是实现 VisionFM 本地模型推理与阿里云百炼多模态大模型的整合，构建完整的 AI 智能分析后端服务。

### 核心成果

- ✅ 创建百炼 API 客户端（`bailian_client.py`）
- ✅ 创建分析整合服务（`analysis_service.py`）
- ✅ 修改推理服务以支持特征提取
- ✅ 新增 AI 分析 API 路由（`ai_analysis.py`）
- ✅ 更新配置管理和主程序
- ✅ 完整的 API 端点和文档

---

## 二、任务完成清单

### 第二阶段计划任务

- [x] 创建 `backend/services/bailian_client.py` - 百炼 API 客户端
- [x] 创建 `backend/services/analysis_service.py` - 分析整合服务
- [x] 修改 `backend/inference_service.py` - 提取结构化特征
- [x] 新增 API 端点 `POST /api/ai/analyze`
- [x] 实现图片 Base64 编码和大小优化
- [x] 设计 Prompt 模板
- [x] 更新 `config.py` 配置管理
- [x] 更新 `main.py` 注册路���

---

## 三、新增文件详解

### 1. `backend/services/__init__.py`

**用途**：服务模块初始化，导出公共接口

```python
from .bailian_client import BaiLianClient
from .analysis_service import AnalysisService

__all__ = ['BaiLianClient', 'AnalysisService']
```

---

### 2. `backend/services/bailian_client.py`

**用途**：阿里云百炼多模态 API 客户端

**核心功能**：

| 方法 | 功能 |
|------|------|
| `__init__(api_key, model)` | 初始化客户端，设置 API Key 和模型 |
| `analyze_fundus_image(...)` | 分析眼底图像，支持原图 + 叠加图 + Prompt |
| `test_connection()` | 测试 API 连通性 |

**关键特性**：

- 支持多张图片输入（原始图 + 分割叠加图）
- 自动提取 token 使用量和响应时间
- 完整的错误处理和日志记录
- 兼容不同的模型选择（qwen-vl-max/plus/chat）

**示例调用**：

```python
from services.bailian_client import BaiLianClient

client = BaiLianClient(api_key="sk-xxx", model="qwen-vl-max")

result = client.analyze_fundus_image(
    image_base64="...",  # Base64 编码的图像
    prompt="请分析这张眼底照片...",
    overlay_base64="...",  # 可选的分割叠加图
    temperature=0.7
)

print(result['content'])  # AI 生成的分析文本
print(result['total_tokens'])  # Token 使用量
```

---

### 3. `backend/services/analysis_service.py`

**用途**：AI 分析整合服务，协调本地推理 + 云端分析

**核心功能**：

| 方法 | 功能 |
|------|------|
| `analyze_image(...)` | 综合分析眼底图像（主入口） |
| `_extract_features(...)` | 从 VisionFM 推理结果提取结构化特征 |
| `_compute_segmentation_features(...)` | 计算分割掩码的形态学特征 |
| `_encode_and_compress_image(...)` | 图像压缩和 Base64 编码 |
| `_create_overlay_image(...)` | 生成分割叠加图 |
| `_build_prompt(...)` | 根据特征生成分析 Prompt |
| `clear_cache()` | 清空分析结果缓存 |

**分析流程**：

```
输入图像 
  ↓
提取结构化特征（分类概率、血管密度等）
  ↓
图像压缩和编码
  ↓
生成分割叠加图
  ↓
构建 Prompt（包含结构化数据）
  ↓
调用百炼 API
  ↓
返回综合报告
```

**特征提取示例**：

```python
features = {
    "classification": {
        "probability": 0.87,
        "predicted_class": 1,
        "class_label": "疾病",
        "confidence": 0.87
    },
    "segmentation": {
        "vessel_area_ratio": 0.15,
        "vessel_density": 0.42,
        "tortuosity_index": 1.23,
        "vessel_pixels": 12345,
        "total_pixels": 262144
    },
    "model_info": {
        "name": "VisionFM",
        "modality": "Fundus",
        "encoder": "ViT-Base"
    }
}
```

**缓存机制**：

- 基于图像 MD5 哈希的内存缓存
- 最多保存 100 条记录
- 超过上限时自动清除最早的 50% 记录

---

### 4. `backend/routers/__init__.py`

**用途**：路由模块初始化

---

### 5. `backend/routers/ai_analysis.py`

**用途**：AI 智能分析路由，提供 HTTP API 端点

**API 端点**：

#### `POST /api/ai/analyze`

上传眼底图像，获取综合分析报告

**请求参数**：

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | ✓ | - | 眼底图像文件（JPG/PNG） |
| `run_segmentation` | bool | ✗ | True | 是否运行分割模型 |
| `run_classification` | bool | ✗ | True | 是否运行分类模型 |
| `seg_checkpoint` | str | ✗ | `checkpoints/seg/checkpoint_108_linear.pth` | 分割模型路径 |
| `cls_checkpoint` | str | ✗ | `checkpoints/single_cls/checkpoint_teacher_linear.pth` | 分类模型路径 |
| `seg_threshold` | float | ✗ | 0.5 | 分割阈值 |
| `temperature` | float | ✗ | 0.7 | AI 创意度（0-1） |

**响应示例**：

```json
{
  "success": true,
  "data": {
    "model_results": {
      "classification": {
        "probability": 0.87,
        "predicted_class": 1,
        "class_label": "疾病",
        "confidence": 0.87
      },
      "segmentation": {
        "vessel_area_ratio": 0.15,
        "vessel_density": 0.42,
        "tortuosity_index": 1.23
      },
      "model_info": {
        "name": "VisionFM",
        "modality": "Fundus",
        "encoder": "ViT-Base"
      }
    },
    "ai_analysis": {
      "content": "【影像所见描述】\n视网膜血管分布清晰...\n\n【异常发现分析】\n...",
      "model_used": "qwen-vl-max"
    },
    "metadata": {
      "image_hash": "abc123def456",
      "timestamp": "2026-02-28T10:30:00",
      "tokens_used": 2048,
      "input_tokens": 1200,
      "output_tokens": 848,
      "api_response_time_ms": 2500,
      "total_time_ms": 3200
    },
    "images": {
      "original": "data:image/jpeg;base64,...",
      "mask": "data:image/png;base64,..."
    },
    "from_cache": false
  }
}
```

#### `GET /api/ai/status`

检查 AI 分析服务状态

**响应示例**：

```json
{
  "enabled": true,
  "api_key_configured": true,
  "model": "qwen-vl-max",
  "cache_size": 5
}
```

#### `POST /api/ai/test-connection`

测试百炼 API 连通性

**响应示例**：

```json
{
  "success": true,
  "message": "API 连接正常",
  "model": "qwen-vl-max"
}
```

#### `POST /api/ai/clear-cache`

清空分析结果缓存

**响应示例**：

```json
{
  "success": true,
  "message": "缓存已清空"
}
```

---

## 四、修改文件详解

### 1. `backend/inference_service.py`

**改动**：`predict_segmentation` 方法返回值新增 `probability_map` 字段

**之前**：
```python
return {
    'mask': mask_bytes.getvalue(),
    'shape': (input_size, input_size),
    'threshold': threshold
}
```

**之后**：
```python
return {
    'mask': mask_bytes.getvalue(),
    'probability_map': prob_np,  # 新增：原始概率图
    'shape': (input_size, input_size),
    'threshold': threshold
}
```

**用途**：支持分析服务计算更精细的分割特征（如血管弯曲度指数）

---

### 2. `backend/main.py`

**改动**：

1. **导入 AI 路由**：
   ```python
   from routers.ai_analysis import router as ai_router
   ```

2. **注册路由**：
   ```python
   app.include_router(ai_router)
   ```

3. **版本升级**：
   ```python
   version="2.1.0"  # 从 2.0.0 升级
   ```

4. **更新根路径响应**：
   - 新增 `"ai_analysis": "AI 智能分析（VisionFM + 多模态大模型）"`
   - 新增 `"ai_analyze": "/api/ai/analyze"` 端点

5. **更新任务列表**：
   - 新增 AI 分析任务描述

6. **启动消息**：
   ```
   支持任务: 分割、二分类、多分类、AI 智能分析
   ```

---

### 3. `backend/config.py`

**改动**：

1. **添加 dotenv 支持**：
   ```python
   from dotenv import load_dotenv
   
   load_dotenv()                      # backend/.env
   load_dotenv(dotenv_path='../.env')  # 项目根目录 .env
   ```

2. **环境变量读取**：
   ```python
   AI_MODEL_TEMPERATURE: float = float(os.getenv('AI_MODEL_TEMPERATURE', '0.7'))
   AI_MAX_TOKENS: int = int(os.getenv('AI_MAX_TOKENS', '2048'))
   ```

**用途**：支持通过 `.env` 文件配置 API Key 和模型参数

---

## 五、Prompt 模板

### 眼底图像分析 Prompt

```
你是一位专业的眼科影像分析助手，具有丰富的眼底疾病诊断经验。

【任务】
请分析这张眼底照片，并结合 VisionFM 模型的检测结果给出专业意见。

【模型检测结果】
- 疾病概率：87.0%
- 预测类别：疾病
- 模型置信度：87.0%
- 血管面积比：0.15
- 血管密度：0.42
- 血管弯曲度指数：1.23

【输出要求】
请按以下结构输出分析结果：

1. **影像所见描述**
   - 描述图像中可见的主要结构（视盘、血管等）
   - 描述分割模型识别出的血管分布特征

2. **异常发现分析**
   - 结合模型预测概率分析疾病风险
   - 说明模型检测到的形态学特征（如血管密度、弯曲度等）

3. **风险评估**
   - 根据概率给出风险等级（低/中/高）
   - 说明模型置信度

4. **进一步检查建议**
   - 建议的后续检查项目
   - 建议的随访时间

请用专业但易懂的语言回答，供医生参考。

【免责声明】
请在分析最后附上：本分析由 AI 辅助生成，仅供医生参考，不作为最终诊断依据。
```

---

## 六、使用指南

### 前置条件

1. **安装依赖**：
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **配置 API Key**：
   
   在项目根目录 `.env` 文件中设置：
   ```env
   DASHSCOPE_API_KEY=sk-your-api-key-here
   DASHSCOPE_MODEL=qwen-vl-max
   ```

3. **验证配置**：
   ```bash
   python -c "from config import config; config.print_config()"
   ```

### 启动后端服务

```bash
cd backend
python main.py
```

输出示例：
```
============================================================
     VisionFM 多任务 API 服务 v2.1
============================================================
     服务地址: http://0.0.0.0:8000
     API 文档: http://0.0.0.0:8000/docs
     支持任务: 分割、二分类、多分类、AI 智能分析
============================================================
```

### 测试 API

#### 1. 检查服务状态

```bash
curl http://localhost:8000/api/ai/status
```

#### 2. 测试 API 连通性

```bash
curl -X POST http://localhost:8000/api/ai/test-connection
```

#### 3. 上传图像进行分析

```bash
curl -X POST http://localhost:8000/api/ai/analyze \
  -F "file=@/path/to/fundus_image.jpg" \
  -F "run_segmentation=true" \
  -F "run_classification=true" \
  -F "temperature=0.7"
```

#### 4. 使用 Python 客户端

```python
import requests

url = "http://localhost:8000/api/ai/analyze"
files = {"file": open("fundus_image.jpg", "rb")}
data = {
    "run_segmentation": True,
    "run_classification": True,
    "temperature": 0.7
}

response = requests.post(url, files=files, data=data)
result = response.json()

print("AI 分析结果：")
print(result['data']['ai_analysis']['content'])
print(f"\nToken 使用量：{result['data']['metadata']['tokens_used']}")
print(f"总耗时：{result['data']['metadata']['total_time_ms']}ms")
```

---

## 七、成本估算

### 单次分析成本

假设：
- 图片编码后约 1500 tokens
- 输入文字约 800 tokens
- AI 输出约 800 tokens

| 模型 | 单次成本 | 说明 |
|------|----------|------|
| qwen-vl-max | ¥0.012 | 最强模型，推荐用于生产环境 |
| qwen-vl-plus | ¥0.005 | 性价比高，推荐用于高并发 |
| qwen-vl-chat | ¥0.003 | 最便宜，推荐用于测试 |

### 月度成本估算

| 使用量 | qwen-vl-max | qwen-vl-plus |
|--------|-------------|--------------|
| 100 次/月 | ¥1.2 | ¥0.5 |
| 1000 次/月 | ¥12 | ¥5 |
| 10000 次/月 | ¥120 | ¥50 |

---

## 八、故障排查

### 问题 1：API Key 未设置

**错误信息**：
```
HTTPException: AI 分析服务未配置：请设置 DASHSCOPE_API_KEY 环境变量
```

**解决方案**：
1. 检查 `.env` 文件是否存在
2. 确认 `DASHSCOPE_API_KEY` 已正确设置
3. 重启后端服务

### 问题 2：API 连接失败

**错误信息**：
```
RuntimeError: 百炼 API 调用失败: ...
```

**解决方案**：
1. 检查网络连接
2. 验证 API Key 有效性
3. 运行 `POST /api/ai/test-connection` 测试连通性
4. 查看后端日志获取详细错误信息

### 问题 3：内存不足

**症状**：处理大图像时内存溢出

**解决方案**：
- 图像自动压缩到 1024x1024，可在 `analysis_service.py` 中调整 `max_size` 参数
- 减小 JPEG 压缩质量（默认 85%）

### 问题 4：缓存过大

**症状**：长时间运行后内存占用持续增长

**解决方案**：
- 调用 `POST /api/ai/clear-cache` 清空缓存
- 或在 `analysis_service.py` 中调整 `_cache_max_size`

---

## 九、后续优化方向

### 短期（第三阶段）

- [ ] 前端 UI 集成（AIAnalysisPanel.vue）
- [ ] 结果缓存持久化（Redis）
- [ ] 批量处理接口
- [ ] 异步任务队列

### 中期

- [ ] 多模型对比分析
- [ ] 医生反馈学习
- [ ] 分析结果导出（PDF/Word）
- [ ] 患者数据管理

### 长期

- [ ] 私有化部署选项（本地 LLM）
- [ ] 多语言支持
- [ ] 移动端 API
- [ ] 医学知识库集成

---

## 十、文件清单

### 新增文件

```
backend/
├── services/
│   ├── __init__.py                    # 服务模块初始化
│   ├── bailian_client.py              # 百炼 API 客户端 (145 行)
│   └── analysis_service.py            # 分析整合服务 (450+ 行)
└── routers/
    ├── __init__.py                    # 路由模块初始化
    └── ai_analysis.py                 # AI 分析路由 (200+ 行)
```

### 修改文件

```
backend/
├── main.py                            # 注册路由，版本升级
├── config.py                          # 添加 dotenv 支持
└── inference_service.py               # 返回值新增 probability_map
```

---

## 十一、版本信息

| 组件 | 版本 |
|------|------|
| VisionFM API | 2.1.0 |
| Python | 3.8+ |
| FastAPI | 0.104.0+ |
| DashScope SDK | 1.14.0+ |
| PyTorch | 1.11.0+ |

---

## 十二、参考资源

- [阿里云百炼官网](https://bailian.console.aliyun.com/)
- [通义千问 VL 文档](https://help.aliyun.com/document_detail/2781831.html)
- [DashScope SDK 文档](https://help.aliyun.com/document_detail/2587497.html)
- [VisionFM 论文](https://ai.nejm.org/doi/full/10.1056/AIoa2300221)

---

*文档版本：v1.0*  
*最后更新：2026-02-28*  
*作者：AI 编码智能体*
