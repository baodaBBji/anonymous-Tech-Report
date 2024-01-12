from data_processing import load_data, preprocess_data
from flashlight_strategy import flashlight_strategy
from tqdm import tqdm
import argparse
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from load_data import load_data

def main(dataset):
    # 定义数据文件路径7
    X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(dataset,
                                                                                         data_dir=data_dir)
    X, y = np.array(X), np.array(y)


    def run_lof(X, y, num_outliers=560, k=60):
        clf = LocalOutlierFactor(n_neighbors=k)
        clf.fit(X)
        lof_scores = -clf.negative_outlier_factor_
        threshold = np.sort(lof_scores)[::-1][num_outliers]
        lof_predictions = np.array(lof_scores > threshold)
        lof_predictions = np.array([int(i) for i in lof_predictions])
        return lof_predictions, lof_scores


    lof_predictions, lof_scores = run_lof(X, y, k=lof_krange[7], num_outliers=int(np.sum(y)))
    
    file_path = 'pendigits.csv'

    # 加载和预处理数据
    raw_data = load_data(file_path)
    data = preprocess_data(raw_data)

    # 定义样本大小和正类标签的索引
    # 如果样本集是数据集的子集，这里可以随机选择或指定样本
    # 假设样本是数据集的前100行
    sample = data[:int(0.1*len(data))]  # 根据具体需求调整样本大小
    positive_index = -1  # 根据数据集结构调整正类标签的位置

    # 执行Flashlight策略
    # def flashlight_strategy(sample, data, positive_index, max_patterns=None, min_info_gain=None, min_f1=0.8):
    explanation_table_df = flashlight_strategy(sample, data, positive_index, max_patterns=448, min_f1=0.8)


    # 输出解释表
    print(explanation_table_df)


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    # 根据实际路径修改
    data_dir = r'../../data/'
    parser = argparse.ArgumentParser(description="Description of your script.")

    # 添加参数
    parser.add_argument("dataset", help="Select the desired dataset", default="Pendigits")

    # 解析命令行参数
    args = parser.parse_args()
    main(args.dataset)
