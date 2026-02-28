# VisionFM AI 分析系统 - 第一阶段实施指南

## 概述

本指南将帮助你完成 AI 分析系统集成的第一阶段：环境准备。

## 第一阶段任务清单

- [ ] 注册阿里云账号并开通百炼服务
- [ ] 获取 API Key
- [ ] 充值测试金额（建议充值 10-20 元用于测试）
- [ ] 安装 Python SDK：`pip install dashscope`
- [ ] 验证 API 连通性

## 详细步骤

### 步骤 1：注册阿里云账号并开通百炼服务

1. **访问阿里云官网**
   - 打开浏览器，访问：https://www.aliyun.com/

2. **注册账号**
   - 如果没有阿里云账号，点击"免费注册"
   - 完成手机号验证和实名认证

3. **开通百炼服务**
   - 访问：https://bailian.console.aliyun.com/
   - 点击"开通服务"
   - 选择"多模态"服务类型

### 步骤 2：获取 API Key

1. **进入 API Key 管理**
   - 在百炼控制台左侧菜单点击"API Key 管理"
   - 或直接访问：https://bailian.console.aliyun.com/api-key

2. **创建 API Key**
   - 点击"创建 API Key"
   - 填写应用名称（如：VisionFM-Analysis）
   - 选择权限：勾选"多模态对话"
   - 点击"确定"

3. **复制 API Key**
   - 创建成功后，复制生成的 API Key
   - ⚠️ **重要**：API Key 只显示一次，请妥善保存

### 步骤 3：充值测试金额

1. **查看账户余额**
   - 在百炼控制台右上角查看账户余额

2. **充值**
   - 点击"充值"按钮
   - 建议首次充值 **10-20 元** 用于测试
   - 支持支付宝、微信、银行卡等支付方式

3. **了解定价**
   ```
   qwen-vl-chat（推荐测试）：¥0.001/千输入tokens + ¥0.002/千输出tokens
   qwen-vl-plus（推荐生产）：¥0.0015/千输入tokens + ¥0.003/千输出tokens
   qwen-vl-max（旗舰模型）：¥0.003/千输入tokens + ¥0.006/千输出tokens
   ```

### 步骤 4：安装 Python SDK

```bash
# 激活项目环境（如果使用 conda）
conda activate vfm

# 或者激活虚拟环境
# source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# 安装 dashscope SDK 和环境变量管理工具
pip install dashscope>=1.14.0 python-dotenv>=1.0.0

# 验证安装
python -c "import dashscope; print('DashScope SDK 安装成功')"
```

### 步骤 5：配置 API Key

1. **复制环境变量模板**
   ```bash
   cp .env.example .env
   ```

2. **编辑 .env 文件，替换 API Key**
   ```bash
   # Windows
   notepad .env

   # Linux/Mac
   nano .env
   ```

   编辑后的内容：
   ```
   DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxx  # 替换为你的真实 API Key
   DASHSCOPE_MODEL=qwen-vl-chat
   ```

### 步骤 6：验证 API 连通性

1. **运行测试脚本**
   ```bash
   python test_bailian.py
   ```

   > 提示：脚本会自动从 `.env` 文件中读取 API Key，无需手动设置环境变量

2. **预期输出**
   ```
   🚀 阿里云百炼 API 连通性测试
   ==================================================
   🔄 正在测试 API 连通性...
   ✅ API 调用成功！
   📝 回复内容：[AI的回复内容]
   🔢 Token 使用量：XXX
   💰 预估成本：¥0.XXXX
   🎉 环境准备完成！可以开始第二阶段开发了。
   ```

3. **如果测试失败**
   - 检查 `.env` 文件中的 API Key 是否正确
   - 检查网络连接
   - 检查账户余额是否充足
   - 查看错误信息并排查

## 故障排除

### 常见问题

1. **API Key 无效**
   ```
   错误：API 调用失败：Invalid API Key
   解决：检查 API Key 是否正确复制
   ```

2. **账户余额不足**
   ```
   错误：API 调用失败：Insufficient balance
   解决：前往百炼控制台充值
   ```

3. **网络连接问题**
   ```
   错误：Connection timeout
   解决：检查网络连接，可能需要代理
   ```

4. **SDK 未安装**
   ```
   错误：ModuleNotFoundError: No module named 'dashscope'
   解决：运行 pip install dashscope
   ```

## 下一步

完成第一阶段后，你可以：

1. **开始第二阶段**：后端开发
   - 创建 `backend/services/bailian_client.py`
   - 创建 `backend/services/analysis_service.py`

2. **查看完整计划**：`docs/plan/ai_analysis_plan.md`

## 技术支持

- 阿里云百炼文档：https://help.aliyun.com/document_detail/2781831.html
- DashScope SDK 文档：https://help.aliyun.com/document_detail/2587497.html
- VisionFM 项目文档：`docs/DEPLOYMENT.md`

---

*第一阶段完成标志：测试脚本运行成功，显示"环境准备完成！"* 