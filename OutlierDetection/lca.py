def is_approx_equal(a, b, threshold=0.1, epsilon=1e-6):
    """判断两个数值是否在给定阈值内近似相等"""
    diff = abs(a - b)
    max_val = max(abs(a), abs(b), epsilon)
    return diff / max_val <= threshold


def lca(t1, t2, threshold=0.1, epsilon=1e-6):
    """计算两个元组的最低共同祖先(LCA)模式，使用阈值进行泛化"""
    lca_pattern = []
    for a, b in zip(t1[:-1], t2[:-1]):  # 不包括类标签
        if isinstance(a, str) or isinstance(b, str):
             lca_pattern.append(a if a == b else '*')
        else:
            if is_approx_equal(a, b, threshold, epsilon):
                lca_pattern.append((a + b) / 2)
            else:
                lca_pattern.append('*')
    return tuple(lca_pattern)
