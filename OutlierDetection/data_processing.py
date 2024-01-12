import csv


def load_data(filepath):
    """从给定路径加载CSV格式的数据"""
    data = []
    with open(filepath, newline='') as file:
        reader = csv.reader(file)
        next(reader, None)  # 跳过头部（标题行）
        for row in reader:
            # 检查每行是否有足够的列
            if len(row) < 3:  # 至少需要3列，包括数据和二元结果
                continue  # 如果不符合，跳过这一行
            try:
                # 检查最后一列是否为 'yes' 或 'no'，并据此转换为 1 或 0
                last_value = row[-1].lower()
                if last_value == 'yes':
                    binary_result = 1
                elif last_value == 'no':
                    binary_result = 0
                else:
                    # 如果最后一列不是 'yes' 或 'no'，假设它是数字并直接转换
                    binary_result = float(row[-1])

                # 将数值特征转换为浮点数
                row_data = [float(value) for value in row[:-1]] + [binary_result]
                # print(f"Converted row: {row_data}")
                data.append(row_data)
            except ValueError as e:
                print(f"ValueError in row {row}: {e}")
                continue  # 如果转换失败，跳过这一行
    return data


def preprocess_data(data):
    """预处理数据"""
    # 这里可以根据需要添加任何数据预处理步骤
    # 当前代码假设数据已经预处理并且可以直接使用
    return data
