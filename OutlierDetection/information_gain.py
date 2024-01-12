import math
from lca import is_approx_equal


def calculate_support(pattern, dataset):
    """计算数据集中模式的支持度"""
    return sum(1 for row in dataset if all(
        attr == '*' or attr == str(value) or (isinstance(value, (int, float)) and is_approx_equal(float(attr), value))
        for attr, value in zip(pattern, row[:-1])
    ))


def calculate_sum_v_u(pattern, dataset, positive_index):
    """计算模式的SUM(v)和SUM(u*)"""
    sum_v = sum(1 for row in dataset if
                all(attr == '*' or attr == str(value) or (
                            isinstance(value, (int, float)) and is_approx_equal(float(attr), value))
                    for attr, value in zip(pattern, row[:-1])) and row[positive_index] == 1)

    sum_u = sum(row[positive_index] for row in dataset if
                all(attr == '*' or attr == str(value) or (
                            isinstance(value, (int, float)) and is_approx_equal(float(attr), value))
                    for attr, value in zip(pattern, row[:-1])))

    return sum_v, sum_u


def information_gain(pattern, dataset, positive_index):
    # 计算模式在数据集中的支持度以及v和u*值的总和
    support = calculate_support(pattern, dataset)
    sum_v, sum_u = calculate_sum_v_u(pattern, dataset, positive_index)

    # 如果没有支持度，信息增益为0
    if support == 0:
        return 0

    # 计算fD,v
    fD_v = sum_v / support if support != 0 else 0

    # 计算gD,u，这里需要一个额外的方法来估计它，暂时使用整个数据集中正例的比例
    total_positive = sum(1 for row in dataset if row[positive_index] == 1)
    gD_u = total_positive / len(dataset) if len(dataset) > 0 else 0


    # 信息增益计算
    gainD = 0
    if gD_u > 0 and fD_v > 0:
        term1 = fD_v * math.log(fD_v / gD_u) if gD_u != fD_v else 0
        term2 = (1 - fD_v) * math.log((1 - fD_v) / (1 - gD_u)) if gD_u != 1 and fD_v != 1 else 0
        gainD = support * (term1 + term2)

        # print(f"fD_v: {fD_v}, gD_u: {gD_u}, support: {support}")
        # print(f"term1: {term1}, term2: {term2}")

    return gainD

