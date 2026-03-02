#!/usr/bin/env python3
"""测试 AI 分析功能"""

import requests

url = 'http://localhost:8000/api/ai/analyze'
files = {'file': open('backend/示例图片.png', 'rb')}
data = {
    'run_segmentation': True,
    'run_classification': True,
    'temperature': 0.7
}

print('正在发送请求...')
response = requests.post(url, files=files, data=data)
result = response.json()

if result.get('success'):
    print('\n=== AI 分析结果 ===')
    print(result['data']['ai_analysis']['content'])
    print('\n=== 模型检测结果 ===')
    cls = result['data']['model_results']['classification']
    seg = result['data']['model_results']['segmentation']
    print(f'疾病概率: {cls["probability"]:.2%}')
    print(f'预测类别: {cls["class_label"]}')
    print(f'血管面积比: {seg["vessel_area_ratio"]:.4f}')
    print(f'血管密度: {seg["vessel_density"]:.4f}')
else:
    print('Error:', result)
