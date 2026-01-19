"""
通义千问API服务
用于生成AI分析报告
"""
import os
from typing import Optional, Dict
import dashscope
from dashscope import Generation


class LLMService:
    """大语言模型服务，用于生成医疗分析报告"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化LLM服务

        Args:
            api_key: 通义千问API Key，如果不提供则从环境变量读取
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            print("⚠️ 警告: 未设置DASHSCOPE_API_KEY，AI报告功能将不可用")
            print("  请设置环境变量或在初始化时传入api_key参数")
        else:
            dashscope.api_key = self.api_key

    def generate_report(self, dr_result: Dict, seg_result: Dict, model: str = "qwen-turbo") -> str:
        """
        生成AI分析报告

        Args:
            dr_result: DR分类结果字典
            seg_result: 分割结果字典
            model: 使用的模型 (qwen-turbo, qwen-plus, qwen-max)

        Returns:
            str: 生成的报告文本
        """
        if not self.api_key:
            return self._get_fallback_report(dr_result, seg_result)

        # 构建prompt
        prompt = self._build_prompt(dr_result, seg_result)

        try:
            # 调用通义千问API
            response = Generation.call(
                model=model,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7,
            )

            if response.status_code == 200:
                return response.output.text
            else:
                print(f"API调用失败: {response.code} - {response.message}")
                return self._get_fallback_report(dr_result, seg_result)

        except Exception as e:
            print(f"生成AI报告时出错: {str(e)}")
            return self._get_fallback_report(dr_result, seg_result)

    def _build_prompt(self, dr_result: Dict, seg_result: Dict) -> str:
        """构建Prompt"""
        # 确定风险等级
        if dr_result['class'] == 1:
            if dr_result['probability'] > 0.8:
                risk_level = "高风险"
            elif dr_result['probability'] > 0.65:
                risk_level = "中风险"
            else:
                risk_level = "低风险"
        else:
            risk_level = "极低风险"

        prompt = f"""你是一位专业的眼科医生助理。请根据以下AI眼底图像分析结果，生成一份详细、专业的医疗分析报告。

## AI分析结果

### DR（糖尿病视网膜病变）分类预测
- 预测类别：{dr_result['class_name']}
- 患病概率：{dr_result['probability_percent']}
- 预测置信度：{dr_result['confidence']}

### 病变区域分割
- 病变面积：{seg_result['lesion_area']:,} 像素
- 病变占比：{seg_result['lesion_ratio_str']}

## 报告要求

请生成包含以下内容的医疗分析报告（使用Markdown格式）：

1. **风险评估** - 根据AI结果评估患者的风险等级（{risk_level}）

2. **主要发现** - 详细解读AI分析结果，包括DR状态和病变情况

3. **建议措施** - 根据风险等级提供具体的医疗建议和生活指导

4. **随访建议** - 提供合理的复查时间安排

## 注意事项

- 报告应该专业、准确、易于理解
- 对于高风险情况，要强调及时就医的重要性
- 报告结尾必须包含免责声明：本报告由AI系统生成，仅供参考，不能替代专业医生诊断
- 使用友好的语言，避免引起过度恐慌

请生成报告："""

        return prompt

    def _get_fallback_report(self, dr_result: Dict, seg_result: Dict) -> str:
        """当API不可用时返回默认报告"""
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
            risk_level = "极低风险"
            risk_emoji = "🟢"

        report = f"""# 眼底图像AI分析报告

## {risk_emoji} 风险评估

| 项目 | 结果 |
|------|------|
| **风险等级** | {risk_level} |
| **DR状态** | {dr_result['class_name']} |
| **患病概率** | {dr_result['probability_percent']} |
| **预测置信度** | {dr_result['confidence']} |

## 🔍 主要发现

### 1. DR分类结果
AI深度学习模型分析显示：
- 该眼底图像预测为：**{dr_result['class_name']}**
- 患病概率为 **{dr_result['probability_percent']}**
- 预测置信度为 **{dr_result['confidence']}**

### 2. 病变区域分析
- 检测到的病变区域面积约为 **{seg_result['lesion_area']:,} 像素**
- 病变区域占图像总面积的 **{seg_result['lesion_ratio_str']}**

## 💡 建议措施

{'### ⚠️ 重要提醒' if dr_result['class'] == 1 else '### ✅ 健康建议'}

{'建议您尽快前往专业眼科医院进行进一步检查，包括但不限于：' if dr_result['class'] == 1 else '建议保持良好的眼部健康习惯：'}

{'- 散瞳眼底检查' if dr_result['class'] == 1 else '- 定期进行眼部检查'}
{'- 眼底荧光血管造影（FFA）' if dr_result['class'] == 1 else '- 保持良好的用眼习惯'}
{'- 光学相干断层扫描（OCT）' if dr_result['class'] == 1 else '- 注意眼部休息'}
{'- 血糖、血压等相关指标检测' if dr_result['class'] == 1 else '- 均衡饮食，适量运动'}

### 通用健康建议
- **血糖控制**：保持血糖在正常范围内，避免血糖波动
- **血压管理**：控制血压，高血压会加重眼底病变
- **血脂控制**：保持血脂正常，预防血管硬化
- **戒烟限酒**：吸烟和过量饮酒会加重眼部病变
- **适度运动**：适量运动有助于改善微循环

## 📅 随访建议

{'建议在 **1-3个月** 内进行复查，并咨询专业眼科医生意见。' if dr_result['class'] == 1 else '建议 **每年** 进行一次常规眼底检查，保持良好的生活习惯。'}

{'复查项目建议：' if dr_result['class'] == 1 else '预防性检查建议：'}
{'- 视力检查' if dr_result['class'] == 1 else '- 视力筛查'}
{'- 眼压检查' if dr_result['class'] == 1 else '- 眼压测量'}
{'- 眼底照相' if dr_result['class'] == 1 else '- 眼底照相'}
{'- 必要时进行FFA或OCT检查' if dr_result['class'] == 1 else ''}

## ⚠️ 免责声明

> 本报告由AI系统自动生成，仅供参考，不能替代专业医生的诊断和治疗方案。
>
> 如有视力变化、眼前黑影、闪光感等症状，请立即就医。
>
> 本系统不承担因使用本报告而产生的任何医疗责任。

---

*报告生成时间：AI辅助系统*
*系统版本：VisionFM DR Analysis System v1.0*
"""

        return report

    def generate_report_simple(self, dr_result: Dict, seg_result: Dict) -> str:
        """
        生成简化版报告（不调用API）

        Args:
            dr_result: DR分类结果字典
            seg_result: 分割结果字典

        Returns:
            str: 报告文本
        """
        return self._get_fallback_report(dr_result, seg_result)


# 创建全局实例（可选）
_llm_service_instance = None

def get_llm_service(api_key: Optional[str] = None) -> LLMService:
    """获取LLM服务实例（单例模式）"""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService(api_key)
    return _llm_service_instance


# 测试代码
if __name__ == '__main__':
    # 测试用例
    dr_test = {
        'class': 1,
        'class_name': 'DR阳性',
        'probability': 0.85,
        'probability_percent': '85.0%',
        'confidence': '高'
    }

    seg_test = {
        'lesion_area': 15000,
        'lesion_ratio_str': '5.73%'
    }

    # 不使用API生成报告
    llm = LLMService()
    report = llm.generate_report_simple(dr_test, seg_test)
    print(report)
