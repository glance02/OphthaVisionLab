# VisionFM AI 分析系统集成方案

> 方案类型：API 调用多模态大模型
> 创建日期：2025-01-10
> 更新日期：2026-03-02
> 状态：后端已完成，前端待开发

---

## 一、方案概述

### 核心思路

将 VisionFM 的本地模型（分割模型、二分类模型）作为"前端特征提取器"，多模态大语言模型作为"后端分析大脑"。

```
输入图像 → VisionFM本地推理 → 提取结构化结果 → 发送给阿里云百炼API → AI生成自然语言分析
```

### 为什么选择这个方案？

| 优势 | 说明 |
|------|------|
| 开发量最小 | 复用现有 VisionFM 后端，2-3天可完成基础版本 |
| AI理解能力强 | 多模态LLM已见过大量医学图像 |
| 无需微调 | 通过Prompt工程即可达到良好效果 |
| 无需本地GPU | 大模型推理在云端完成 |
| 灵活可调 | 可随时切换更好的模型或调整Prompt |

---

## 二、系统架构

### 整体流程图

```
┌─────────────────────────────────────────────────────────────┐
│  第1步：用户上传图像                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         患者的眼底照片                               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  第2步：VisionFM 本地模型推理                                 │
│  ┌──────────────────────┐  ┌──────────────────────────┐    │
│  │  二分类模型           │  │   分割模型               │    │
│  │  → 概率: 0.87         │  │   → 血管掩码             │    │
│  │  → 类别: 疾病         │  │   → 形态学特征           │    │
│  └──────────────────────┘  └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  第3步：准备发送给API的内容                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • 原始眼底照片                                      │   │
│  │  • 分割掩码叠加图                                    │   │
│  │  • 结构化数据（分类概率、分割特征）                  │   │
│  │  • Prompt（分析任务描述）                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  第4步：调用阿里云百炼多模态API                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  阿里云百炼 API:                                     │   │
│  │  • 通义千问-VL-Max（推荐，中文医学理解能力强）       │   │
│  │  • 通义千问-VL-Plus（性价比高）                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  第5步：AI 返回分析结果                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  自然语言描述：                                      │   │
│  │  • 影像所见描述                                      │   │
│  │  • 异常发现说明                                      │   │
│  │  • 诊断建议                                          │   │
│  │  • 进一步检查建议                                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 与 VisionFM 现有架构集成

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (Vue 3)                          │
│                    http://localhost:5173                    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ ImageUpload  │  │ResultDisplay │  │ AIAnalysisPanel  │  │
│  │   图片上传    │  │  分割结果展示 │  │  AI 分析结果面板  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  后端 (FastAPI)                              │
│                    http://localhost:8000                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  POST /api/segment          # 现有：图像分割         │  │
│  │  POST /api/ai/analyze       # 新增：AI智能分析       │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌─────────────────────────┼──────────────────────────┐    │
│  │                         │                          │    │
│  ▼                         ▼                          ▼    │
│ ┌─────────────┐      ┌─────────────┐      ┌─────────────┐  │
│ │  SegService │      │  ClsService │      │  BaiLian    │  │
│ │  分割模型    │      │  分类模型    │      │  API Client │  │
│ └─────────────┘      └─────────────┘      └─────────────┘  │
│       │                     │                     │         │
│       └─────────────────────┴─────────────────────┘         │
│                           │                                 │
│                    ┌─────────────┐                         │
│                    │  Analysis   │                         │
│                    │  Service    │                         │
│                    │ 分析整合服务 │                         │
│                    └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、技术选型

### API 服务：阿里云百炼

| 特性 | 说明 |
|------|------|
| 访问速度 | 国内访问快，阿里云基础设施稳定 |
| 模型选择 | 支持通义千问 VL 系列等多模态模型 |
| 成本 | 比直接调用 GPT-4V 便宜很多 |
| API格式 | 兼容 OpenAI API 格式，易于集成 |
| 医学理解 | 通义千问在中文医学领域表现出色 |

### 推荐模型

| 模型 | 特点 | 推荐场景 |
|------|------|----------|
| qwen-vl-max | 通义千问视觉版旗舰模型，中文医学理解最强 | 首选 |
| qwen-vl-plus | 视觉版增强模型，性价比高 | 高并发场景 |
| qwen-vl-chat-v1 | 基础视觉对话模型，成本最低 | 默认使用 |
| qwen-vl-chat | 兼容旧版本 | 兼容 |

### SDK 选择

```bash
# 阿里云 dashscope SDK（推荐）
pip install dashscope python-dotenv

```

---

## 四、实施计划

### 第一阶段：环境准备（0.5天）

#### 任务清单

- [x] 注册阿里云账号并开通百炼服务
- [x] 获取 API Key
- [x] 充值测试金额（建议充值 10-20 元用于测试）
- [x] 安装 Python SDK：`pip install dashscope`
- [x] 验证 API 连通性（参考下方的测试脚本）

### 第二阶段：后端开发（已完成）

#### 任务清单

- [x] 创建 `backend/config.py` - 配置管理（含 API Key）
- [x] 创建 `backend/services/bailian_client.py` - 百炼 API 客户端
- [x] 创建 `backend/services/analysis_service.py` - 分析整合服务
- [x] 创建 `backend/routers/ai_analysis.py` - AI分析路由
- [x] 修改 `backend/main.py` - 集成 AI 路由（版本升级至 2.1.0）
- [x] 实现图片 Base64 编码和大小优化
- [x] 设计 Prompt 模板
- [x] 实现内存缓存机制

#### 已实现功能

| 文件 | 功能说明 |
|------|----------|
| `config.py` | DASHSCOPE_API_KEY、DASHSCOPE_MODEL 等配置 |
| `bailian_client.py` | 百炼 MultiModalConversation API 封装，支持多图输入 |
| `analysis_service.py` | 分析整合服务，包含特征提取、图像处理、Prompt构建、缓存管理 |
| `ai_analysis.py` | API 路由：`/api/ai/analyze`、`/api/ai/status`、`/api/ai/test-connection` |

### 第三阶段：前端开发（待开发）

#### 任务清单

- [ ] 创建 `frontend/src/components/AIAnalysisPanel.vue` - AI分析结果面板
- [ ] 修改前端组件添加分析按钮
- [ ] 添加 API 调用服务
- [ ] 实现分析结果展示（Markdown 渲染）

### 第四阶段：测试优化

#### 任务清单

- [ ] 准备多张测试图片
- [ ] 调试 Prompt 优化输出质量
- [ ] 调整图片大小（平衡质量和成本）
- [ ] 记录调用成本和响应时间

---

## 五、核心功能设计

### 5.1 发送给API的内容

**图片部分：**
- 原始眼底照片（压缩后，建议 512x512 或 1024x1024）
- 分割掩码叠加图（让AI看到识别区域）
- 可选：热力图、对比图

**文字数据（JSON格式）：**
```json
{
  "classification": {
    "probability": 0.87,
    "class": "疾病",
    "threshold": 0.5
  },
  "segmentation": {
    "vessel_area_ratio": 0.15,
    "vessel_density": 0.42,
    "tortuosity_index": 1.23,
    "optic_disc_detected": true
  },
  "model_info": {
    "name": "VisionFM",
    "modality": "Fundus",
    "version": "1.0"
  }
}
```

### 5.2 Prompt 模板设计

```
你是一位专业的眼科影像分析助手，具有丰富的眼底疾病诊断经验。

【任务】
请分析这张眼底照片，并结合 VisionFM 模型的检测结果给出专业意见。

【模型检测结果】
- 疾病概率：{probability}%
- 预测类别：{class}
- 血管面积比：{vessel_area_ratio}
- 血管密度：{vessel_density}
- 血管弯曲度指数：{tortuosity_index}
- 视盘检测：{optic_disc_detected}

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
```

### 5.3 输出报告格式

```json
{
  "success": true,
  "data": {
    "model_results": {
      "classification": {
        "probability": 0.87,
        "predicted_class": 1,
        "class_label": "疾病",
        "confidence": 0.95
      },
      "segmentation": {
        "vessel_area_ratio": 0.15,
        "vessel_density": 0.42,
        "tortuosity_index": 1.23,
        "vessel_pixels": 38000,
        "total_pixels": 262144
      },
      "model_info": {
        "name": "VisionFM",
        "modality": "Fundus",
        "encoder": "ViT-Base"
      }
    },

    "ai_analysis": {
      "content": "完整的AI分析文本（Markdown格式）...",
      "model_used": "qwen-vl-chat-v1"
    },

    "metadata": {
      "image_hash": "a1b2c3d4...",
      "timestamp": "2026-03-02T10:30:00",
      "tokens_used": 2048,
      "input_tokens": 1500,
      "output_tokens": 548,
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

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_results` | object | VisionFM 本地模型的结构化检测数据 |
| `model_results.classification` | object | 分类结果（概率、类别、置信度） |
| `model_results.segmentation` | object | 分割特征（血管面积比、密度、弯曲度） |
| `ai_analysis.content` | string | AI 生成的分析文本（Markdown格式） |
| `metadata` | object | 调用元信息（tokens、耗时、缓存状态） |
| `images` | object | 原始图像和分割掩码（base64，供前端展示） |
| `from_cache` | bool | 是否命中缓存 |

---

## 六、项目结构

### 新增文件

```
VisionFM/
│
├── backend/                      # 后端服务
│   ├── config.py                 # 配置管理（含 DASHSCOPE_API_KEY）
│   ├── services/
│   │   ├── __init__.py
│   │   ├── bailian_client.py     # 阿里云百炼 API 客户端 ✅ 已完成
│   │   └── analysis_service.py   # 分析整合服务 ✅ 已完成
│   ├── routers/
│   │   ├── __init__.py
│   │   └── ai_analysis.py        # AI分析路由 ✅ 已完成
│   ├── inference_service.py      # 现有：模型推理服务
│   └── main.py                   # 已集成 AI 路由
│
├── frontend/                     # 前端应用
│   └── src/
│       ├── components/
│       │   └── AIAnalysisPanel.vue   # AI分析结果面板 ⏳ 待开发
│       └── services/
│           └── aiAnalysis.js     # AI分析 API 调用 ⏳ 待开发
│
└── docs/
    └── plan/
        └── ai_analysis_plan.md   # 本文档
```

### 配置文件

在项目根目录或 `backend/` 目录下创建 `.env` 文件：

```bash
# 阿里云百炼配置
DASHSCOPE_API_KEY=your-api-key-here
DASHSCOPE_MODEL=qwen-vl-chat-v1  # 可选，默认 qwen-vl-chat-v1

# 可选配置
AI_MODEL_TEMPERATURE=0.7
AI_MAX_TOKENS=2048
```

### 关键代码示例

#### 1. 配置文件 (backend/config.py)

```python
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件

class Config:
    DASHSCOPE_API_KEY: str = os.getenv('DASHSCOPE_API_KEY')
    DASHSCOPE_MODEL: str = os.getenv('DASHSCOPE_MODEL', 'qwen-vl-chat-v1')
    AI_ANALYSIS_ENABLED: bool = True
    MAX_FILE_SIZE_MB: int = 10

config = Config()
```

#### 2. 百炼 API 客户端 (backend/services/bailian_client.py) - 已实现

核心功能：
- 支持同时输入原图和分割叠加图
- 自动提取 token 使用量
- 返回响应时间统计

```python
class BaiLianClient:
    def __init__(self, api_key: str, model: str = "qwen-vl-max"):
        dashscope.api_key = api_key
        self.model = model

    def analyze_fundus_image(
        self,
        image_base64: str,
        prompt: str,
        overlay_base64: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict:
        # 构建多图输入 messages
        content = [{"image": f"data:image/jpeg;base64,{image_base64}"}]
        if overlay_base64:
            content.append({"image": f"data:image/png;base64,{overlay_base64}"})
        content.append({"text": prompt})

        messages = [{"role": "user", "content": content}]
        response = MultiModalConversation.call(...)

        return {
            "content": ai_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "model": self.model,
            "response_time_ms": elapsed_ms,
        }
```

#### 3. 分析服务 (backend/services/analysis_service.py) - 已实现

核心功能：
- 从分割掩码计算形态学特征（血管面积比、密度、弯曲度）
- 图像压缩和叠加图生成
- 简易内存缓存（MD5 哈希）
- Prompt 模板自动构建

```python
class AnalysisService:
    async def analyze_image(
        self,
        image_bytes: bytes,
        seg_result: Optional[dict] = None,
        cls_result: Optional[dict] = None,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> dict:
        # 1. 缓存检查（MD5 哈希）
        # 2. 提取结构化特征
        # 3. 图片压缩 + Base64 编码
        # 4. 生成叠加图
        # 5. 构建 Prompt
        # 6. 调用百炼 API
        # 7. 返回结果（含 metadata）
```

#### 4. API 端点 (backend/routers/ai_analysis.py) - 已实现

```python
router = APIRouter(prefix="/api/ai", tags=["AI Analysis"])

@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    run_segmentation: bool = Form(True),
    run_classification: bool = Form(True),
    seg_checkpoint: str = Form("checkpoints/seg/checkpoint_108_linear.pth"),
    cls_checkpoint: str = Form("checkpoints/single_cls/checkpoint_teacher_linear.pth"),
    temperature: float = Form(0.7),
):
    # 1. VisionFM 本地推理（分割 + 分类）
    # 2. 调用分析服务
    # 3. 返回结果 + 原图/掩码 base64
```

**已实现的 API 端点：**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/ai/analyze` | POST | AI 智能分析主接口 |
| `/api/ai/status` | GET | 检查服务状态 |
| `/api/ai/test-connection` | POST | 测试 API 连通性 |
| `/api/ai/clear-cache` | POST | 清空分析缓存 |

---

## 七、成本估算

### 阿里云百炼定价（参考）

| 模型 | 输入价格 | 输出价格 |
|------|----------|----------|
| qwen-vl-max | ¥0.003/千 tokens | ¥0.006/千 tokens |
| qwen-vl-plus | ¥0.0015/千 tokens | ¥0.003/千 tokens |
| qwen-vl-chat | ¥0.001/千 tokens | ¥0.002/千 tokens |

> 注：图片按 tokens 计费，约 512x512 的图片 ≈ 1000-2000 tokens

### 单次分析成本估算

```
假设：
- 图片编码后约 1500 tokens (512x512)
- 输入文字约 800 tokens
- AI输出约 800 tokens

使用 qwen-vl-max：
单次成本 = (1500+800) × 0.003/1000 + 800 × 0.006/1000 ≈ ¥0.012

使用 qwen-vl-plus（性价比之选）：
单次成本 = (1500+800) × 0.0015/1000 + 800 × 0.003/1000 ≈ ¥0.005
```

### 月度成本估算

| 使用量 | qwen-vl-max | qwen-vl-plus |
|--------|-------------|--------------|
| 100次/月 | ¥1.2 | ¥0.5 |
| 1000次/月 | ¥12 | ¥5 |
| 10000次/月 | ¥120 | ¥50 |

---

## 八、注意事项

### 8.1 数据隐私与合规

- ⚠️ **患者图像和数据会发送到阿里云百炼 API**
- **建议措施**：
  - 对患者信息进行脱敏处理（去除姓名、身份证号等）
  - 仅上传医学图像，不上传患者个人信息
  - 如果是医院使用，需确认数据合规要求
  - 可考虑添加用户协议，明确告知数据处理方式

### 8.2 API 限制

- 有并发请求限制（根据账户等级）
- 有速率限制（默认 QPS 限制）
- 建议实现：
  - 请求队列
  - 重试机制（指数退避）
  - 超时处理

### 8.3 图像处理

- **尺寸限制**：建议压缩至 512x512 或 1024x1024 以减少 tokens
- **格式**：JPEG（有损压缩，更小）或 PNG（无损，更清晰）
- **质量平衡**：JPEG quality 85% 是较好的平衡点

### 8.4 责任声明

- AI分析结果仅供参考，**不能替代医生诊断**
- 在使用界面应加入免责声明：
  > "本分析由 AI 辅助生成，仅供医生参考，不作为最终诊断依据。如有疑问，请咨询专业医生。"

---

## 九、后续优化方向

1. **结果缓存**：
   - 相同图像使用 MD5 哈希缓存结果
   - Redis 缓存常用分析结果

2. **批量处理**：
   - 异步队列处理多张图片
   - 批量 API 调用降低单位成本

3. **多模型对比**：
   - 同时调用 qwen-vl-max 和 qwen-vl-plus
   - 对比分析结果，选择置信度高的

4. **反馈学习**：
   - 记录医生对 AI 分析的修改意见
   - 用于优化 Prompt 和模型选择

5. **私有化部署（可选）**：
   - 如对数据隐私要求极高，可考虑部署本地视觉语言模型
   - 如 Qwen-VL 本地部署（需要 GPU）

---

## 十、参考链接

- 阿里云百炼官网：https://bailian.console.aliyun.com/
- 通义千问 VL 文档：https://help.aliyun.com/document_detail/2781831.html
- DashScope SDK 文档：https://help.aliyun.com/document_detail/2587497.html
- VisionFM 后端文档：`docs/DEPLOYMENT.md`

---

## 附录：API 测试脚本

项目根目录已提供测试脚本 `test_bailian.py`：

```bash
# 运行测试
python test_bailian.py

# 或设置环境变量后运行
export DASHSCOPE_API_KEY=your-api-key-here
python test_bailian.py
```

测试脚本会：
1. 检查 API Key 是否配置
2. 发送简单的文本请求测试连通性
3. 估算 Token 使用量和成本

**API 端点测试：**

后端服务启动后，可使用以下方式测试：

```bash
# 测试服务状态
curl http://localhost:8000/api/ai/status

# 测试 API 连通性
curl -X POST http://localhost:8000/api/ai/test-connection

# 完整分析（需要图片）
curl -X POST http://localhost:8000/api/ai/analyze \
  -F "file=@test_fundus.jpg" \
  -F "run_segmentation=true" \
  -F "run_classification=true"
```

---

*文档版本：v2.1*
*最后更新：2026-03-02*
*更新说明：后端实现已完成，标记为"后端已完成，前端待开发"状态*
