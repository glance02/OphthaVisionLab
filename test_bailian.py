# test_bailian.py - 阿里云百炼 API 连通性测试脚本

import dashscope
from dashscope import MultiModalConversation
import base64
import os
from pathlib import Path
from dotenv import load_dotenv

# 自动加载 .env 文件
load_dotenv()

def test_bailian_api():
    """
    测试阿里云百炼 API 连通性
    """
    # 从 .env 文件或环境变量获取 API Key
    api_key = os.getenv('DASHSCOPE_API_KEY')

    if not api_key:
        print("❌ 错误：未找到 DASHSCOPE_API_KEY 环境变量")
        print("请设置环境变量：")
        print("Windows: set DASHSCOPE_API_KEY=your_api_key_here")
        print("Linux/Mac: export DASHSCOPE_API_KEY=your_api_key_here")
        return False

    # 设置 API Key
    dashscope.api_key = api_key

    print("🔄 正在测试 API 连通性...")

    try:
        # 准备测试消息（使用文字测试，避免需要图片）
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": "你好，请简单介绍一下你自己。"}
                ]
            }
        ]

        # 调用 API
        response = MultiModalConversation.call(
            model="qwen-vl-chat-v1",  # 使用基础模型测试，成本最低
            messages=messages,
            temperature=0.7
        )

        if response.status_code == 200:
            print("✅ API 调用成功！")
            print(f"📝 回复内容：{response.output.choices[0].message.content}")
            
            # 尝试获取 token 信息（不同模型可能返回格式不同）
            try:
                if hasattr(response.usage, 'total_tokens'):
                    total_tokens = response.usage.total_tokens
                    print(f"🔢 Token 使用量：{total_tokens}")
                    
                    # 估算成本（qwen-vl-chat-v1: 输入 ¥0.001/千tokens, 输出 ¥0.002/千tokens）
                    if hasattr(response.usage, 'input_tokens') and hasattr(response.usage, 'output_tokens'):
                        input_cost = response.usage.input_tokens * 0.001 / 1000
                        output_cost = response.usage.output_tokens * 0.002 / 1000
                        total_cost = input_cost + output_cost
                        print(f"💰 估算成本：¥{total_cost:.4f}")
            except:
                pass  # 某些模型可能不返回 token 信息
            
            return True
        else:
            print(f"❌ API 调用失败：{response.message}")
            print(f"状态码：{response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 发生异常：{str(e)}")
        return False

def test_with_image():
    """
    使用测试图片进行完整测试（需要准备测试图片）
    """
    print("\n🔄 正在测试图片分析功能...")

    # 尝试多个可能的图片位置
    possible_paths = [
        Path("backend/示例图片.png"),  # 后端示例图片
        Path("示例图片.png"),
        Path("test_fundus.png"),
        Path("test_fundus.jpg"),
    ]
    
    test_image_path = None
    for path in possible_paths:
        if path.exists():
            test_image_path = path
            print(f"✅ 找到测试图片: {path}")
            break
    
    if not test_image_path:
        print("⚠️  跳过图片测试：未找到测试图片")
        print("   提示：图片分析功能在第二阶段后端实现时会进一步优化")
        return

    try:
        from PIL import Image
        
        # 读取图片信息
        img = Image.open(test_image_path)
        print(f"   图片信息: {img.format} {img.size} {img.mode}")
        
        # 这里仅作演示，实际图片处理会在后端 service 中实现
        print("   ℹ️  Base64 编码图片调用已准备好")
        print("   📝 完整的图片分析功能将在第二阶段后端中实现")
        return True

    except Exception as e:
        print(f"❌ 图片处理异常：{str(e)}")

if __name__ == "__main__":
    print("🚀 阿里云百炼 API 连通性测试")
    print("=" * 50)

    # 基本连通性测试
    success = test_bailian_api()

    if success:
        # 图片测试
        test_with_image()

        print("\n🎉 环境准备完成！可以开始第二阶段开发了。")
    else:
        print("\n❌ 请检查 API Key 和网络连接。")