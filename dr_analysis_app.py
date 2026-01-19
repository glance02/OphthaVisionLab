"""
眼底图像分析系统 - Streamlit Web应用
功能：DR二分类预测、病变区域分割、AI分析报告生成
"""
import streamlit as st
import io
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from model_service import ModelService
from llm_service import get_llm_service

# ==================== 配置 ====================
st.set_page_config(
    page_title="眼底图像分析系统",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive-result {
        background-color: #fff5f5;
        border-left: 5px solid #fc8181;
        padding: 1rem;
        border-radius: 5px;
    }
    .negative-result {
        background-color: #f0fff4;
        border-left: 5px solid #68d391;
        padding: 1rem;
        border-radius: 5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 会话状态初始化 ====================
@st.cache_resource
def load_model_service():
    """缓存模型服务，避免重复加载"""
    # 自动检测是否有GPU，没有则使用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with st.spinner(f"正在加载AI模型（使用设备: {device.upper()}），请稍候..."):
        service = ModelService(device=device)
        time.sleep(0.5)  # 显示加载动画
    return service

def initialize_session_state():
    """初始化会话状态变量"""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'dr_result' not in st.session_state:
        st.session_state.dr_result = None
    if 'seg_result' not in st.session_state:
        st.session_state.seg_result = None
    if 'ai_report' not in st.session_state:
        st.session_state.ai_report = None
    if 'use_llm_api' not in st.session_state:
        st.session_state.use_llm_api = False

initialize_session_state()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("🔧 系统设置")

    st.markdown("---")
    st.markdown("### 📊 模型信息")
    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.info(f"""
    - **DR分类模型**: VisionFM + ClsHead
    - **分割模型**: VisionFM + UNETR
    - **预训练权重**: VFM_Fundus
    - **设备**: {device_info}
    """)

    st.markdown("---")
    st.markdown("### 📖 使用说明")
    st.markdown("""
    1. 上传眼底图像
    2. 点击"开始分析"按钮
    3. 查看AI分析结果
    4. 阅读AI生成的报告
    """)

    st.markdown("---")
    st.markdown("### ⚙️ AI报告设置")
    st.session_state.use_llm_api = st.checkbox(
        "使用通义千问API生成报告",
        value=False,
        help="开启后使用通义千问API生成更专业的报告，需要配置API Key"
    )
    if st.session_state.use_llm_api:
        api_key = st.text_input(
            "API Key",
            type="password",
            help="请输入通义千问API Key"
        )
        if api_key:
            st.success("✅ API Key已设置")
    st.markdown("---")
    st.markdown("### ⚠️ 免责声明")
    st.warning("""
    本系统仅供辅助参考，
    不能替代专业医生诊断。
    如有异常请及时就医！
    """)

# ==================== 主界面 ====================
st.markdown('<p class="main-header">👁️ 眼底图像分析系统</p>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
基于VisionFM深度学习模型的糖尿病视网膜病变(DR)智能辅助诊断系统
</div>
""", unsafe_allow_html=True)

# ==================== 图像上传区域 ====================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader(
        "## 📁 上传眼底图像",
        type=['png', 'jpg', 'jpeg'],
        help="请上传眼底彩照图像（PNG/JPG格式）"
    )

    if uploaded_file is not None:
        # 读取并显示图像
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="上传的图像", use_container_width=True)

        # 分析按钮
        st.markdown("---")
        analyze_button = st.button("🔍 开始分析", type="primary", use_container_width=True)

        if analyze_button:
            with st.spinner("正在进行AI分析，请稍候..."):
                try:
                    # 加载模型（首次）
                    if not st.session_state.model_loaded:
                        model_service = load_model_service()
                        st.session_state.model_service = model_service
                        st.session_state.model_loaded = True
                    else:
                        model_service = st.session_state.model_service

                    # 执行分析
                    result = model_service.predict_combined(image)

                    # 保存结果
                    st.session_state.dr_result = result['dr']
                    st.session_state.seg_result = result['segmentation']

                    # 生成AI报告
                    llm_service = get_llm_service(api_key if st.session_state.use_llm_api and api_key else None)
                    if st.session_state.use_llm_api and api_key:
                        with st.spinner("正在生成AI报告..."):
                            st.session_state.ai_report = llm_service.generate_report(
                                result['dr'],
                                result['segmentation']
                            )
                    else:
                        st.session_state.ai_report = llm_service.generate_report_simple(
                            result['dr'],
                            result['segmentation']
                        )

                    st.session_state.analysis_done = True

                    st.success("✅ 分析完成！")

                except Exception as e:
                    st.error(f"分析失败：{str(e)}")
                    st.session_state.analysis_done = False

# ==================== 结果展示区域 ====================
if st.session_state.analysis_done:
    st.markdown("---")
    st.markdown("## 📋 分析结果")

    # 创建结果展示的列布局
    result_col1, result_col2 = st.columns(2)

    # === DR分类结果 ===
    with result_col1:
        st.markdown("### 🔬 DR预测结果")

        dr_result = st.session_state.dr_result

        # 根据预测结果显示不同的样式
        if dr_result['class'] == 1:
            st.markdown(f"""
            <div class="positive-result">
                <h3 style='color: #c53030;'>⚠️ DR阳性</h3>
                <p style='font-size: 1.2rem;'><strong>患病概率: {dr_result['probability_percent']}</strong></p>
                <p>置信度: <strong>{dr_result['confidence']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="negative-result">
                <h3 style='color: #2f855a;'>✅ 健康</h3>
                <p style='font-size: 1.2rem;'><strong>健康概率: {dr_result['probability_percent']}</strong></p>
                <p>置信度: <strong>{dr_result['confidence']}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        # 显示指标
        st.metric("预测类别", dr_result['class_name'])
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("概率值", f"{dr_result['probability']:.4f}")
        with col_metric2:
            st.metric("置信度", dr_result['confidence'])

    # === 分割结果 ===
    with result_col2:
        st.markdown("### 🎨 病变区域分割")

        seg_result = st.session_state.seg_result

        # 显示分割掩码
        st.image(seg_result['mask'], caption="病变区域分割结果", use_container_width=True)

        # 显示指标
        col_metric3, col_metric4 = st.columns(2)
        with col_metric3:
            st.metric("病变面积", f"{seg_result['lesion_area']:,} px²")
        with col_metric4:
            st.metric("病变占比", seg_result['lesion_ratio_str'])

    # ==================== AI分析报告 ====================
    st.markdown("---")
    st.markdown("## 📝 AI分析报告")

    # 使用预生成的AI报告
    if st.session_state.ai_report:
        st.markdown(st.session_state.ai_report)
    else:
        # 备用报告生成逻辑
        dr_result = st.session_state.dr_result
        seg_result = st.session_state.seg_result

        # 确定风险等级
        if dr_result['class'] == 1:
            if dr_result['probability'] > 0.8:
                risk_level = "高风险"
                risk_emoji = "🔴"
            elif dr_result['probability'] > 0.65:
                risk_level = "中风险"
                risk_emoji = "🟡"
            else:
                risk_level = "低风险"
                risk_emoji = "🟢"
        else:
            if dr_result['probability'] < 0.2:
                risk_level = "极低风险"
                risk_emoji = "🟢"
            else:
                risk_level = "低风险"
                risk_emoji = "🟢"

        # 生成报告
        report = f"""
---

### {risk_emoji} 风险评估

| 项目 | 结果 |
|------|------|
| **风险等级** | {risk_level} |
| **DR状态** | {dr_result['class_name']} |
| **患病概率** | {dr_result['probability_percent']} |
| **预测置信度** | {dr_result['confidence']} |

---

### 🔍 主要发现

1. **DR分类结果**
   - AI模型预测该眼底图像为：**{dr_result['class_name']}**
   - 患病概率为 **{dr_result['probability_percent']}**，预测置信度为 **{dr_result['confidence']}**

2. **病变区域分析**
   - 检测到的病变区域面积约为 **{seg_result['lesion_area']:,} 像素**
   - 病变区域占图像总面积的 **{seg_result['lesion_ratio_str']}**

---

### 💡 建议措施

{'<span style="color: #c53030;">⚠️ 建议尽快就医，进行进一步的专业眼科检查</span>' if dr_result['class'] == 1 else '<span style="color: #2f855a;">✅ 眼底图像未见明显异常，建议定期检查</span>'}

**一般建议：**
- 保持良好的血糖控制
- 定期进行眼底检查
- 注意眼部卫生，避免过度用眼
- 如有视力变化，及时就医

---

### 📅 随访建议

{'建议在 **1-3个月** 内进行复查，并咨询专业眼科医生意见。' if dr_result['class'] == 1 else '建议 **每年** 进行一次常规眼底检查，保持良好的生活习惯。'}

---

> ⚠️ **重要提示**：本报告由AI系统自动生成，仅供参考，不能替代专业医生的诊断。如有任何疑虑，请及时咨询专业眼科医生。

---

*报告生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}*
*系统版本：VisionFM DR Analysis System v1.0*
"""

        st.markdown(report)

    # 下载按钮
    st.markdown("---")
    col_download1, col_download2, col_download3 = st.columns([2, 2, 2])

    with col_download2:
        if st.button("📥 下载分析报告", use_container_width=True):
            st.info("💡 提示：请使用浏览器的打印功能（Ctrl+P）保存为PDF")

# ==================== 页脚 ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.8rem;'>
<p>© 2024 眼底图像分析系统 | 基于VisionFM深度学习模型</p>
<p>本项目仅用于科研和教学演示，不作为临床诊断依据</p>
</div>
""", unsafe_allow_html=True)

# ==================== 示例图片提示 ====================
if uploaded_file is None:
    st.markdown("---")
    st.info("💡 **提示**：您可以上传眼底彩照图像进行AI分析。支持PNG、JPG、JPEG格式。")
