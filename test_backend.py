"""
后端 API 测试脚本
测试分割功能并保存结果
"""
import requests
import base64
from pathlib import Path
from PIL import Image
import io


def test_segmentation_api(image_path: str, output_dir: str = "test_results"):
    """测试分割 API"""
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # API 端点
    url = "http://localhost:8000/api/segment"

    # 准备文件
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/png')}
        data = {
            'checkpoint': 'checkpoints/checkpoint_108_linear.pth',
            'threshold': '0.5'
        }

        print(f"📤 上传图像: {image_path}")
        print(f"🔗 API 端点: {url}")
        print("-" * 60)

        # 发送请求
        response = requests.post(url, files=files, data=data, timeout=60)

        # 检查响应
        if response.status_code == 200:
            result = response.json()

            if result.get('success'):
                print("✅ 分割成功！")
                print(f"   任务类型: {result['task']}")
                print(f"   阈值: {result['data']['threshold']}")
                print(f"   形状: {result['data']['shape']}")
                print()

                # 解码并保存原图
                original_data = result['data']['originalImage'].split(',', 1)[1]
                original_bytes = base64.b64decode(original_data)
                original_img = Image.open(io.BytesIO(original_bytes))

                original_output = output_path / f"{Path(image_path).stem}_original.png"
                original_img.save(original_output)
                print(f"💾 原图已保存: {original_output}")

                # 解码并保存掩码
                mask_data = result['data']['maskImage'].split(',', 1)[1]
                mask_bytes = base64.b64decode(mask_data)
                mask_img = Image.open(io.BytesIO(mask_bytes))

                mask_output = output_path / f"{Path(image_path).stem}_mask.png"
                mask_img.save(mask_output)
                print(f"💾 掩码已保存: {mask_output}")

                print()
                print("=" * 60)
                print("🎉 测试完成！")
                print(f"   原图大小: {original_img.size}")
                print(f"   掩码大小: {mask_img.size}")
                print(f"   输出目录: {output_path.absolute()}")
                print("=" * 60)

                return True
            else:
                print(f"❌ 分割失败: {result}")
                return False
        else:
            print(f"❌ HTTP 错误: {response.status_code}")
            print(f"   响应: {response.text}")
            return False


if __name__ == "__main__":
    import sys

    # 测试图像路径
    image_path = sys.argv[1] if len(sys.argv) > 1 else "1.png"

    if not Path(image_path).exists():
        print(f"❌ 图像文件不存在: {image_path}")
        sys.exit(1)

    # 运行测试
    success = test_segmentation_api(image_path)

    sys.exit(0 if success else 1)
