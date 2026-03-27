import os
from pathlib import Path

def convert_multiclass_to_binary(input_file):
    """
    将多分类标签文件转换为二分类标签文件（原地修改）

    参数:
        input_file: 要修改的标签文件路径
    """
    # 读取所有行并转换
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                lines.append('')
                continue

            # 解析行: 文件路径;标签
            parts = line.split(';')
            if len(parts) != 2:
                print(f"跳过无效行: {line}")
                lines.append('')
                continue

            file_path, label = parts
            try:
                label_int = int(label)
                # 转换规则:
                # 0 (Healthy) -> 0 (Healthy)
                # 1, 2, 3, 4 (DR-1, DR-2, DR-3, DR-4) -> 1 (DR)
                if label_int == 0:
                    new_label = 0  # Healthy保持为0
                elif label_int in [1, 2, 3, 4]:
                    new_label = 1  # 所有DR级别转换为1
                else:
                    print(f"未知标签: {label_int}，跳过")
                    lines.append('')
                    continue

                # 保存转换后的行
                lines.append(f"{file_path};{new_label}")
            except ValueError:
                print(f"无法解析标签: {label}，跳过")
                lines.append('')
                continue

    # 原地写入文件
    with open(input_file, 'w') as f_out:
        for line in lines:
            if line:  # 只写入非空行
                f_out.write(line + '\n')

def process_fundus_classification():
    """
    处理FundusClassification/IDRiD/目录下的标签文件（原地修改）
    """
    base_dir = "dataset/SingleModalCls/FundusClassification/IDRiD"

    # 处理训练集标签文件
    train_input = os.path.join(base_dir, "training_labels.txt")
    print(f"处理训练集标签文件: {train_input}")
    convert_multiclass_to_binary(train_input)

    # 处理测试集标签文件
    test_input = os.path.join(base_dir, "test_labels.txt")
    print(f"处理测试集标签文件: {test_input}")
    convert_multiclass_to_binary(test_input)


def verify_conversion():
    """
    验证转换结果
    """
    base_dir = "dataset/SingleModalCls/FundusClassification/IDRiD"

    # 检查训练集
    train_file = os.path.join(base_dir, "training_labels.txt")
    if os.path.exists(train_file):
        healthy_count = 0
        dr_count = 0

        with open(train_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(';')
                if len(parts) == 2:
                    try:
                        label = int(parts[1])
                        if label == 0:
                            healthy_count += 1
                        elif label == 1:
                            dr_count += 1
                    except ValueError:
                        pass

        print(f"训练集验证: Healthy={healthy_count}, DR={dr_count}, 总计={healthy_count+dr_count}")

    # 检查测试集
    test_file = os.path.join(base_dir, "test_labels.txt")
    if os.path.exists(test_file):
        healthy_count = 0
        dr_count = 0

        with open(test_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(';')
                if len(parts) == 2:
                    try:
                        label = int(parts[1])
                        if label == 0:
                            healthy_count += 1
                        elif label == 1:
                            dr_count += 1
                    except ValueError:
                        pass

        print(f"测试集验证: Healthy={healthy_count}, DR={dr_count}, 总计={healthy_count+dr_count}")

if __name__ == "__main__":
    print("开始将FundusClassification数据集从多分类转换为二分类...")
    process_fundus_classification()
    verify_conversion()
    print("转换完成!")