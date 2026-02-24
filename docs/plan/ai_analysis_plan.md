# VisionFM AI 分析系统集成方案

> 方案类型：API 调用多模态大模型
> 创建日期：2025-01-10
> 更新日期：2026-02-24
> 状态：待实施

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
| qwen-vl-chat | 基础视觉对话模型，成本最低 | 测试/开发阶段 |

### SDK 选择

```bash
# 方案1: 阿里云 dashscope SDK（推荐）
pip install dashscope

# 方案2: OpenAI 兼容模式
pip install openai
```

---

## 四、实施计划

### 第一阶段：环境准备（0.5天）

#### 任务清单

- [ ] 注册阿里云账号并开通百炼服务
- [ ] 获取 API Key
- [ ] 充值测试金额（建议充值 10-20 元用于测试）
- [ ] 安装 Python SDK：`pip install dashscope`
- [ ] 验证 API 连通性（参考下方的测试脚本）

### 第二阶段：后端开发（1.5天）

#### 任务清单

- [ ] 创建 `backend/services/bailian_client.py` - 百炼 API 客户端
- [ ] 创建 `backend/services/analysis_service.py` - 分析整合服务
- [ ] 修改 `backend/inference_service.py` - 提取结构化特征
- [ ] 新增 API 端点 `POST /api/ai/analyze`
- [ ] 实现图片 Base64 编码和大小优化
- [ ] 设计 Prompt 模板

### 第三阶段：前端开发（1天）

#### 任务清单

- [ ] 创建 `frontend/src/components/AIAnalysisPanel.vue` - AI分析结果面板
- [ ] 修改 `frontend/src/components/ResultDisplay.vue` - 添加分析按钮
- [ ] 添加 API 调用服务
- [ ] 实现分析结果展示（Markdown 渲染）

### 第四阶段：测试优化（1天）

#### 任务清单

- [ ] 准备多张测试图片
- [ ] 调试 Prompt 优化输出质量
- [ ] 调整图片大小（平衡质量和成本）
- [ ] 记录调用成本和响应时间
- [ ] 添加缓存机制（相同图像不重复调用）

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
    "patient_id": "P001",
    "image_id": "IMG_20250110_001",
    "timestamp": "2025-01-10T10:30:00",
    
    "model_results": {
      "classification": {
        "probability": 0.87,
        "class": "疾病"
      },
      "segmentation": {
        "vessel_area_ratio": 0.15,
        "vessel_density": 0.42,
        "tortuosity_index": 1.23
      }
    },
    
    "ai_analysis": {
      "findings": "视网膜血管密度异常增高...",
      "abnormalities": "血管弯曲度超出正常范围...",
      "risk_level": "高风险",
      "recommendation": "建议立即进行进一步眼科检查",
      "full_response": "完整的AI分析文本..."
    },
    
    "metadata": {
      "model_used": "qwen-vl-max",
      "tokens_used": 2048,
      "cost_cny": 0.015,
      "response_time_ms": 2500
    }
  }
}
```

---

## 六、项目结构

### 新增文件

```
VisionFM/
│
├── backend/                      # 后端服务
│   ├── services/
│   │   ├── __init__.py
│   │   ├── bailian_client.py     # 阿里云百炼 API 客户端
│   │   └── analysis_service.py   # 分析整合服务
│   ├── routers/
│   │   ├── __init__.py
│   │   └── ai_analysis.py        # AI分析路由
│   └── config.py                 # 配置管理（含 API Key）
│
├── frontend/                     # 前端应用
│   └── src/
│       ├── components/
│       │   └── AIAnalysisPanel.vue   # AI分析结果面板
│       └── services/
│           └── aiAnalysis.js     # AI分析 API 调用
│
└── docs/
    └── plan/
        └── ai_analysis_plan.md   # 本文档
```

### 关键代码示例

#### 1. 百炼 API 客户端 (backend/services/bailian_client.py)

```python
import dashscope
from dashscope import MultiModalConversation

class BaiLianClient:
    def __init__(self, api_key: str):
        dashscope.api_key = api_key
        self.model = "qwen-vl-max"  # 或 qwen-vl-plus
    
    def analyze_fundus_image(
        self,
        image_base64: str,
        prompt: str,
        temperature: float = 0.7
    ) -> dict:
        """
        分析眼底图像
        
        Args:
            image_base64: Base64 编码的图像
            prompt: 分析提示词
            temperature: 创意度（0-1）
        
        Returns:
            AI 分析结果
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{image_base64}"},
                    {"text": prompt}
                ]
            }
        ]
        
        response = MultiModalConversation.call(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        
        return {
            "content": response.output.choices[0].message.content,
            "tokens": response.usage.total_tokens,
            "model": self.model
        }
```

#### 2. 分析服务 (backend/services/analysis_service.py)

```python
from .bailian_client import BaiLianClient
from ..inference_service import SegmentationService, ClassificationService

class AIService:
    def __init__(
        self,
        bailian_client: BaiLianClient,
        seg_service: SegmentationService,
        cls_service: ClassificationService
    ):
        self.bailian = bailian_client
        self.seg = seg_service
        self.cls = cls_service
    
    async def analyze_image(self, image_bytes: bytes) -> dict:
        # 1. 执行 VisionFM 本地推理
        seg_result = self.seg.predict(image_bytes)
        cls_result = self.cls.predict(image_bytes)
        
        # 2. 提取特征指标
        features = self._extract_features(seg_result, cls_result)
        
        # 3. 生成叠加图
        overlay_image = self._create_overlay(image_bytes, seg_result)
        
        # 4. 准备 Prompt
        prompt = self._build_prompt(features)
        
        # 5. 调用百炼 API
        analysis = self.bailian.analyze_fundus_image(
            image_base64=overlay_image,
            prompt=prompt
        )
        
        return {
            "model_results": features,
            "ai_analysis": analysis
        }
```

#### 3. API 端点 (backend/routers/ai_analysis.py)

```python
from fastapi import APIRouter, UploadFile, File, Depends
from ..services.analysis_service import AIService

router = APIRouter(prefix="/api/ai", tags=["AI Analysis"])

@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    ai_service: AIService = Depends(get_ai_service)
):
    """
    上传眼底图像，获取 AI 分析报告
    """
    image_bytes = await file.read()
    result = await ai_service.analyze_image(image_bytes)
    return {"success": True, "data": result}
```

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

```python
# test_bailian.py
import dashscope
from dashscope import MultiModalConversation

def test_bailian_api():
    # 设置 API Key
    dashscope.api_key = "your-api-key-here"
    
    # 准备测试消息
    messages = [
        {
            "role": "user",
            "content": [
                {"image": "https://example.com/test_fundus.jpg"},
                {"text": "这是一张眼底照片，请描述你看到了什么？"}
            ]
        }
    ]
    
    # 调用 API
    response = MultiModalConversation.call(
        model="qwen-vl-max",
        messages=messages
    )
    
    if response.status_code == 200:
        print("API 调用成功！")
        print(f"回复内容：{response.output.choices[0].message.content}")
        print(f"Token 使用量：{response.usage.total_tokens}")
    else:
        print(f"API 调用失败：{response.message}")

if __name__ == "__main__":
    test_bailian_api()
```

---

*文档版本：v2.0*  
*最后更新：2026-02-24*  
*更新说明：将 API 提供商从 SiliconFlow 更换为阿里云百炼*
