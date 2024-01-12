import math
import sys

import numpy as np
from sklearn import metrics
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier

from load_data import load_data


def cal_f1score(stats_list, return_detail=False):
    if len(stats_list) == 0:
        if return_detail:
            return tuple([0] * 4)
        else:
            return 0

    all_statistics = [[stat.label * stat.num * stat.acc,
                       stat.label * stat.num * (1 - stat.acc),
                       (1 - stat.label) * stat.num * (1 - stat.acc)] for stat in stats_list]
    TP = np.sum(np.array(all_statistics)[:, 0])
    FP = np.sum(np.array(all_statistics)[:, 1])
    FN = np.sum(np.array(all_statistics)[:, 2])
    if return_detail:
        return TP / (TP + (FP + FN) / 2 + 1e-5), TP, FP, FN
    else:
        return TP / (TP + (FP + FN) / 2 + 1e-5)


def entropy(rule):
    p = rule.acc
    ent = - p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
    return 1 - ent


def cal_list_entropy(stats_list, return_detail=False):
    if len(stats_list) == 0:
        if return_detail:
            return tuple([0] * 2)
        else:
            return 0

    entropy_list = np.array([1 - entropy(rule) for rule in stats_list])
    lengths = np.array([rule.num for rule in stats_list])

    total_length = np.sum(lengths)
    avg_entropy = np.sum(entropy_list * lengths / total_length)
    if return_detail:
        return avg_entropy, total_length
    else:
        return avg_entropy


class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.lchild = None
        self.rchild = None
        self.is_leaf = True
        self.split_num = None
        # self.attribute = None

    def set_rule(self, rule):
        self.rule = rule

    def predict_single(self, x):
        if not self.is_leaf:
            if x[self.rule.split_attribute] <= self.rule.split_num:
                return self.lchild.predict_single(x)
            else:
                return self.rchild.predict_single(x)
        else:
            return self.rule.label

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])


def findEntropy(data, rows):
    yes = 0
    no = 0
    ans = -1
    idx = len(data[0]) - 1
    entropy = 0
    for i in rows:
        if data[i][idx] == 'Yes':
            yes = yes + 1
        else:
            no = no + 1

    x = yes / (yes + no)
    y = no / (yes + no)
    if x != 0 and y != 0:
        entropy = -1 * (x * math.log2(x) + y * math.log2(y))
    if x == 1:
        ans = 1
    if y == 1:
        ans = 0
    return entropy, ans


def cal_split_entropy(X, y, cur_rule, col, val):
    idxes_1 = cur_rule.idxes[np.where(X[cur_rule.idxes, col] <= val)]
    idxes_2 = cur_rule.idxes[np.where(X[cur_rule.idxes, col] > val)]

    tmp_y_1 = y[idxes_1]
    tmp_y_2 = y[idxes_2]

    label_1 = int(np.sum(tmp_y_1) > len(tmp_y_1) / 2)
    label_2 = int(np.sum(tmp_y_2) > len(tmp_y_2) / 2)

    assert len(tmp_y_1) > 0
    assert len(tmp_y_2) > 0

    acc_1 = np.sum(tmp_y_1 == label_1) / len(tmp_y_1)
    acc_2 = np.sum(tmp_y_2 == label_2) / len(tmp_y_2)

    new_avg_entropy = len(tmp_y_1) * (1 - (-acc_1 * np.log(acc_1 + 1e-5)
                                           - (1 - acc_1) * np.log(1 - acc_1 + 1e-5))) + len(tmp_y_2) * (
                              1 - (-acc_2 * np.log(acc_2 + 1e-5)
                                   - (1 - acc_2) * np.log(1 - acc_2 + 1e-5)))
    return new_avg_entropy


def findMaxGain(X, y, stats_list, rows, columns, cur_rule):
    maxGain = 0
    best_split_num = None
    rows_smaller = None
    rows_larger = None
    best_attribute = None

    if len(np.unique(y[rows])) == 1:
        return maxGain, best_attribute, best_split_num, rows_smaller, rows_larger

    # len_of_rules = np.sum([len(stat.attributes) for stat in stats_list])

    avg_entropy0, total_length0 = cal_list_entropy(stats_list, return_detail=True)
    avg_entropy = (avg_entropy0 * total_length0 + entropy(cur_rule) * cur_rule.num) / (cur_rule.num + total_length0)
    cur_metric = avg_entropy

    best_entropy = -1000
    for col in columns:

        for val in np.sort(np.unique(X[cur_rule.idxes, col]))[:-1]:

            new_avg_entropy = cal_split_entropy(X, y, cur_rule, col, val)

            if new_avg_entropy > best_entropy:
                best_attribute = col
                maxGain = new_avg_entropy - cur_metric
                best_split_num = val
                rows_smaller = cur_rule.idxes[np.where(X[cur_rule.idxes, col] <= val)]
                rows_larger = cur_rule.idxes[np.where(X[cur_rule.idxes, col] > val)]
                best_entropy = new_avg_entropy

    if maxGain < 0:
        print("maxgain < 0")

    return maxGain, best_attribute, best_split_num, rows_smaller, rows_larger


def buildTree(X, predictions, stats, cur_rule, stats_list, rows, columns):
    '''
    start building the decision tree for data X[rows][columns]
    params:
    X: all data
    predictions: lof predictions
    rows: rows we need to deal with now
    columns: columns we need to deal with now (usually full data columns)
    attributes: array of numbers, the columns/attributes that has been used before
    stats_list: the rules outside of current data X
    '''

    # first need to find the attribute, the split number, the points that are larger or smaller than the split number;
    # to maximize the objective

    maxGain, attribute, split_num, rows_smaller, rows_larger = findMaxGain(X, predictions, stats_list, rows, columns,
                                                                           cur_rule)

    if maxGain == 0:
        # it means we cannot split the current rule further
        root = Node()
        root.is_leaf = True
        root.value = int(np.sum(predictions[rows]) > len(rows) / 2)
        return root

    print("maxGain:", maxGain, "idxes_smaller:", len(rows_smaller), "idxes_larger", len(rows_larger), "rule length",
          len(list(set(cur_rule.attributes + [attribute]))), "split_num:", split_num)

    # split the current rule, the same as splitting the current data X[rows]
    # rows_smaller = idxes_smaller
    # rows_larger = idxes_larger

    label_1 = int(np.sum(predictions[rows_smaller]) > len(rows_smaller) / 2)
    label_2 = int(np.sum(predictions[rows_larger]) > len(rows_larger) / 2)
    acc_1 = np.sum(predictions[rows_smaller] == label_1) / len(rows_smaller)
    acc_2 = np.sum(predictions[rows_larger] == label_2) / len(rows_larger)

    rule_1 = stats(len(rows_smaller), acc_1, label_1, rows_smaller, list(set(cur_rule.attributes + [attribute])))
    rule_2 = stats(len(rows_larger), acc_2, label_2, rows_larger, list(set(cur_rule.attributes + [attribute])))

    print("best attribute:", attribute, "label 1:", label_1, "num 1:", len(rows_smaller), "label 2:", label_2, "num 2:",
          len(rows_larger))

    root = Node()
    root.is_leaf = False
    root.attribute = attribute
    root.split_num = split_num
    # root.childs = []

    root.lchild = buildTree(X, predictions, stats,
                            cur_rule=rule_1,
                            stats_list=stats_list + [rule_2],
                            rows=rows_smaller, columns=columns)
    if root.lchild.is_leaf:
        root.lchild.rule = rule_1

    root.rchild = buildTree(X, predictions, stats,
                            cur_rule=rule_2,
                            stats_list=stats_list + [rule_1],
                            rows=rows_larger, columns=columns)
    if root.rchild.is_leaf:
        root.rchild.rule = rule_2

    return root


def update_attributes_and_intervals(cur_attributes, cur_intervals, attribute, value, mode='left'):
    if attribute in cur_attributes:
        idx = cur_attributes.index(attribute)
        assert cur_intervals[idx][1] >= value and cur_intervals[idx][0] <= value

    else:
        idx = -1
        cur_attributes.append(attribute)
        cur_intervals.append([-np.inf, np.inf])

    if mode == 'left':
        cur_intervals[idx][1] = value
    elif mode == 'right':
        cur_intervals[idx][0] = value


def traverse(root, rules):
    if root.lchild is not None: traverse(root.lchild, rules)
    if root.rchild is not None: traverse(root.rchild, rules)
    if root.lchild is None and root.rchild is None:
        rules.append(root.rule)



def cal_length_entropy_gain(rule, columns, X, y, max_length=15):
    length_divide_entropy = []
    split_attributes = []
    split_nums = []

    assert rule.num == len(rule.idxes)

    entropys = []

    for col in columns:

        if col in rule.attributes:
            incremental_length = len(rule.attributes)
        else:
            incremental_length = len(rule.attributes) + 2
            if len(rule.attributes) + 1 > max_length:
                continue

        for val in np.sort(np.unique(X[rule.idxes, col]))[:-1]:

            new_avg_entropy = cal_split_entropy(X, y, rule, col, val)

            entropy_gain = new_avg_entropy - entropy(rule) * rule.num

            if entropy_gain <= 0: continue

            entropys.append(new_avg_entropy)
            length_divide_entropy.append(incremental_length / entropy_gain)
            split_attributes.append(col)
            split_nums.append(val)

    if len(length_divide_entropy) > 0:
        idx = np.argmin(np.array(length_divide_entropy))
        return length_divide_entropy[idx], split_attributes[idx], split_nums[idx]
    else:
        return tuple([-1] * 3)


def select(input_list, idxes):
    new_list = []
    for i, x in enumerate(input_list):
        if i in idxes:
            new_list.append(x)
    return new_list


def delete(input_list, idxes):
    new_list = []
    for i, x in enumerate(input_list):
        if i not in idxes:
            new_list.append(x)
    return new_list


def expand_rule(X, rule_to_expand, predictions, leaf_list, stats_list, stats, print_result=True):

    if print_result:
        print(f"split rule:, split_num:{rule_to_expand.split_num}, split_attribute: {rule_to_expand.split_attribute}")

    idxes_smaller = rule_to_expand.idxes[
        np.where(X[rule_to_expand.idxes, rule_to_expand.split_attribute] <= rule_to_expand.split_num)]
    idxes_larger = rule_to_expand.idxes[
        np.where(X[rule_to_expand.idxes, rule_to_expand.split_attribute] > rule_to_expand.split_num)]

    label_1 = int(np.sum(predictions[idxes_smaller]) > len(idxes_smaller) / 2)
    label_2 = int(np.sum(predictions[idxes_larger]) > len(idxes_larger) / 2)

    acc_1 = np.sum(predictions[idxes_smaller] == label_1) / len(idxes_smaller)
    acc_2 = np.sum(predictions[idxes_larger] == label_2) / len(idxes_larger)

    rule_1 = stats(len(idxes_smaller), acc_1, label_1, idxes_smaller,
                   list(set(rule_to_expand.attributes + [rule_to_expand.split_attribute])))
    rule_2 = stats(len(idxes_larger), acc_2, label_2, idxes_larger,
                   list(set(rule_to_expand.attributes + [rule_to_expand.split_attribute])))

    leaf_list.append(rule_1)
    leaf_list.append(rule_2)
    stats_list.append(rule_1)
    stats_list.append(rule_2)

    node_1 = Node()
    node_2 = Node()

    node_1.set_rule(rule_1)
    node_2.set_rule(rule_2)
    rule_1.set_node(node_1)
    rule_2.set_node(node_2)

    rule_to_expand.node.lchild = node_1
    rule_to_expand.node.rchild = node_2
    rule_to_expand.node.is_leaf = False


def expand_rule_and_add_to_list(leaf_list, stats_list, stats, rule, cur_C, columns, predictions,
                                max_length, avg_entropy0, total_length0, threshold, print_result=True):
    '''
    leaf_list: current leaf list, rule is not included in this list
    '''
    num_1 = rule.num
    expand_rule(X, rule, predictions, leaf_list, stats_list, stats, print_result=print_result)
    assert leaf_list[-1].num + leaf_list[-2].num == num_1

    new_rules = leaf_list[-2:]
    new_idxes = [len(leaf_list) - 2, len(leaf_list) - 1]

    for rule, rule_idx in zip(new_rules, new_idxes):
        rule.length_divide_entropy, rule.split_attribute, rule.split_num = \
            cal_length_entropy_gain(rule, columns, X, predictions, max_length=max_length)

        if not rule.length_divide_entropy == -1:
            rule_C = rule.length_divide_entropy * avg_entropy0 * total_length0 - sum(
                [len(r.attributes) for r in leaf_list])

            if rule_C < cur_C:
                avg_entropy0, total_length0 = cal_list_entropy(leaf_list, return_detail=True)
                leaf_list = delete(leaf_list, [leaf_list.index(rule)])
                expand_rule_and_add_to_list(leaf_list, stats_list, stats, rule, cur_C, columns, predictions,
                                            max_length, avg_entropy0, total_length0, threshold, print_result=True)

                f1_score_2 = cal_f1score(leaf_list)
                if f1_score_2 > threshold: return


def buildRuleTree(X, predictions, stats, cur_rule, stats_list, leaf_list, rows, columns, max_length, threshold,
                  print_result=True):
    '''
    X: Data Points
    predictions: no need to explain
    stas: namedtuple
    cur_rule: current_rule
    stats_list: rules except for the current rule
    rows: covered by cur_rule
    columns: covered by cur_rule
    '''

    C_along_the_process = []
    rule_num_along_the_process = []

    cur_C = 0

    f1_score = cal_f1score(leaf_list)
    while f1_score < threshold:

        avg_entropy0, total_length0 = cal_list_entropy(leaf_list, return_detail=True)

        cur_idx = -1

        leaf_C = []
        for idx, rule in enumerate(leaf_list):
            if rule.length_divide_entropy is None:
                rule.length_divide_entropy, rule.split_attribute, rule.split_num = \
                    cal_length_entropy_gain(rule, columns, X, predictions, max_length=max_length)

            if rule.length_divide_entropy == -1:
                leaf_C.append(np.inf)
            else:
                leaf_C.append(rule.length_divide_entropy * avg_entropy0 * total_length0 - sum(
                    [len(r.attributes) for r in leaf_list]))

        leaf_C = np.array(leaf_C)
        C_along_the_process.append(cur_C)
        rule_num_along_the_process.append(len(leaf_list))

        if print_result:
            print("C = ", cur_C, "f1 score:", f1_score)

        idxes = np.where(leaf_C <= cur_C)[0]

        if len(idxes) > 0:

            rules_to_expand = select(leaf_list, idxes)
            leaf_list = delete(leaf_list, idxes)

            flag = 0
            for r_i, r in enumerate(rules_to_expand):
                expand_rule(X, r, predictions, leaf_list, stats_list, stats, print_result=print_result)
                f1_score = cal_f1score(leaf_list + rules_to_expand[r_i:])
                if f1_score >= threshold:
                    flag = 1
                    break
            if flag: break

        elif (leaf_C == np.inf).all():
            break

        else:
            idx = np.argmin(leaf_C)
            cur_C = leaf_C[idx]

            # remove idx in leaf_list
            rule_to_expand = leaf_list[idx]
            del leaf_list[idx]

            expand_rule(X, rule_to_expand, predictions, leaf_list, stats_list, stats, print_result=print_result)

        f1_score = cal_f1score(leaf_list)

    if print_result:
        print("Final F-1 score:", cal_f1score(leaf_list))

    C_along_the_process.append(cur_C)
    rule_num_along_the_process.append(len(leaf_list))
    if print_result:
        print(C_along_the_process)
        print(rule_num_along_the_process)

    return leaf_list


def draw_C(rule_num, C, name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    import pandas as pd
    df = pd.DataFrame({"Rule Number": rule_num, "C": C, "color": 'blue'})
    sns.set_theme(style="darkgrid")
    ax = sns.lineplot(x='Rule Number', y='C', data=df, markers=True, marker='o', color='blue')
    plt.savefig(f"../increase_of_C_{name}.png", dpi=600)


def calculate(X, predictions, criterion=1, max_length=10, max_depth=-1, threshold=0.90, print_result=True,
              return_rule_num=False):
    print("Threshold:", threshold)
    class stats:
        def __init__(self, num, acc, label, idxes, attributes):
            self.num = num
            self.acc = acc
            self.label = label
            self.idxes = idxes
            self.attributes = attributes
            self.length_divide_entropy = None
            self.split_attribute = None
            self.split_num = None

        def set_node(self, node):
            self.node = node

        def extra_repr(self):
            return f"num = {self.num}, acc = {self.acc}, label = {self.label}, attributes = {self.attributes}"

    stats_list = []
    rows = np.arange(X.shape[0])
    columns = np.arange(X.shape[1])
    root_rule = stats(len(X), np.sum(predictions == 0) / len(predictions), np.sum(predictions) // len(predictions),
                      rows, [])
    root = Node()
    root_rule.set_node(root)
    root.set_rule(root_rule)
    stats_list.append(root_rule)
    leaf_list = [root_rule]

    leaf_list = buildRuleTree(X, predictions, stats, root_rule, stats_list, leaf_list, rows, columns, max_length,
                              threshold, print_result=print_result)
    if root.is_leaf: root.rule = root_rule

    rules = leaf_list

    if print_result:
        print("rules F-1 score", cal_f1score(rules))
        # print("rules F-1 score", cal_f1score(leaf_list))
        print("whole lengths:", np.sum([len(r.attributes) for r in rules]))
        print('lengths:', [len(r.attributes) for r in rules])
        print("Total Rule number:", len(rules))

    if return_rule_num:
        return root, np.sum([len(r.attributes) for r in rules]), len(rules)
    else:
        return root


def main(name, lof_predictions, lof_scores, ratio, max_length, threshold, method='MDT'):

    # X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(name, data_dir=r'D:\ANow\data/')

    y = np.zeros(len(X))

    if method == 'MDT':
        # Random subsample

        outlier_idxes = np.where(lof_predictions == 1)[0]
        Outliers = X[outlier_idxes]

        idxes = np.where(lof_predictions == 0)[0]
        selected_indices = np.random.choice(np.arange(len(idxes)), size=int(len(idxes) * ratio), replace=False)
        Inliers = X[idxes[selected_indices]]

        X_new = np.concatenate([Outliers, Inliers], axis=0)
        lof_predictions_new = np.concatenate([np.ones(len(Outliers)), np.zeros(len(Inliers))], axis=0)

        import time
        start_time = time.time()
        root, length, num = calculate(X_new, lof_predictions_new, max_length=max_length, criterion=1, print_result=True, threshold=threshold, return_rule_num=True)
        end_time = time.time()

        predictions = root.predict(X)
        f1_score = metrics.f1_score(lof_predictions, predictions)
        time = end_time - start_time
        print("Final F1 score:", f1_score)
        print("Time consumed:", time)
        return f1_score, time, length, num

    elif method == 'CART':
        clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
        path = clf.cost_complexity_pruning_path(X, lof_predictions)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        for ccp_alpha in ccp_alphas[::-1]:
            clf = DecisionTreeClassifier(criterion='entropy', random_state=0, ccp_alpha=ccp_alpha)
            clf.fit(X, lof_predictions)
            if metrics.f1_score(lof_predictions, clf.predict(X)) > threshold:
                break
        leaf_nodes = clf.apply(X)
        lengths = []
        from sklearn.tree import _tree
        def tree_to_code(tree, feature_names):
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
            print("def tree({}):".format(", ".join(feature_names)))

            def recurse(node, depth, attributes):
                indent = "  " * depth
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    # print("{}if {} <= {}:".format(indent, name, threshold))
                    attributes = set(attributes + [name])

                    recurse(tree_.children_left[node], depth + 1, list(attributes))
                    # print("{}else:  # if {} > {}".format(indent, name, threshold))

                    recurse(tree_.children_right[node], depth + 1, list(attributes))
                else:
                    # print("{}return {}".format(indent, tree_.value[node]))
                    lengths.append(len(attributes))

            recurse(0, 1, [])

        tree_to_code(clf, [str(i) for i in np.arange(X.shape[1])])
        # print(lengths)
        f1_score = metrics.f1_score(lof_predictions, clf.predict(X))
        print("final F-1 score:", f1_score)
        print("whole lengths:", np.sum(lengths))
        print("lengths:", lengths)
        print("Total rule Number:", len(lengths))
        return f1_score, 0, np.sum(lengths), len(lengths)

    else:
        for depth in range(7, 25):
            clf = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=depth)
            # clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
            clf.fit(X, lof_predictions)
            leaf_nodes = clf.apply(X)
            lengths = []
            from sklearn.tree import _tree
            def tree_to_code(tree, feature_names):
                tree_ = tree.tree_
                feature_name = [
                    feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature
                ]
                # print("def tree({}):".format(", ".join(feature_names)))

                def recurse(node, depth, attributes):
                    indent = "  " * depth
                    if tree_.feature[node] != _tree.TREE_UNDEFINED:
                        name = feature_name[node]
                        threshold = tree_.threshold[node]
                        # print("{}if {} <= {}:".format(indent, name, threshold))
                        attributes = set(attributes + [name])

                        recurse(tree_.children_left[node], depth + 1, list(attributes))
                        # print("{}else:  # if {} > {}".format(indent, name, threshold))

                        recurse(tree_.children_right[node], depth + 1, list(attributes))
                    else:
                        # print("{}return {}".format(indent, tree_.value[node]))
                        lengths.append(len(attributes))

                recurse(0, 1, [])

            tree_to_code(clf, [str(i) for i in np.arange(X.shape[1])])
            # print(lengths)
            f1_score = metrics.f1_score(lof_predictions, clf.predict(X))
            print(f"Depth: {depth}, f1_score: {f1_score}")
            if f1_score > threshold: break
        print("final F-1 score:", f1_score)
        print("whole lengths:", np.sum(lengths))
        print("lengths:", lengths)
        print("Total rule Number:", len(lengths))
        return f1_score, 0, np.sum(lengths), len(lengths)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'All':
            names = ['Pendigits', 'PageBlock', "Shuttle", "Pima", "Mammography", "Satimage-2", "cover", "satellite",
                     "SpamBase", "Http"]
        else:
            names = [sys.argv[1]]
    else:
        names = ["Pendigits"]

    data_dir = r'../data/'
    # max_lengths = [2, 4, 6, 8, 10, 12]
    max_lengths = [10]
    thresholds = [0.8]
    # thresholds = [0.7, 0.75, 0.8, 0.85, 0.95]
    results = []

    for name in names:

        print(name)

        X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(name,
                                                                                         data_dir=data_dir)

        parts = 10
        # 找到每列的最大值和最小值
        max_values, min_values = find_max_min_values(X)
        row = len(X)  # 可以根据需要修改 n 的值
        column = parts*len(X[0])  # 可以根据需要修改每行的元素个数
        # 使用列表推导创建二维数组
        new_x = [[0 for j in range(column)] for i in range(row)]

        for i in range(len(X[0])):
        # for i in range(len(max_values)):
            result_ranges = split_into_ten_ranges(min_values[i], max_values[i], parts)
            for j in range(len(X)):
                result_vector = number_to_binary_vector(parts,X[j][i], result_ranges)
                for xx in range(len(result_vector)):
                    if result_vector[xx] == 1:
                        new_x[j][parts*i + xx] = 1


        X, y = np.array(X), np.array(y)

        def run_lof(X, y, num_outliers=560, k=60):
            clf = LocalOutlierFactor(n_neighbors=k)
            clf.fit(X)
            lof_scores = -clf.negative_outlier_factor_
            threshold = np.sort(lof_scores)[::-1][num_outliers]
            lof_predictions = np.array(lof_scores > threshold)
            lof_predictions = np.array([int(i) for i in lof_predictions])
            # print("F-1 score:", metrics.f1_score(y, lof_predictions))
            return lof_predictions, lof_scores


        lof_predictions, lof_scores = run_lof(X, y, k=lof_krange[7], num_outliers=int(np.sum(y)))


        C = CorelsClassifier(max_card=2, c=0.8, n_iter=100000, verbosity=["loud"])

        new_x = np.array(new_x)

        # Create the model, with 10000 as the maximum number of iterations 
        # c = CorelsClassifier(c=0.8,n_iter=10000)

        # Fit, and score the model on the training set
        # a = c.fit(new_x, lof_predictions).score(new_x, lof_predictions)

        # Fit, and score the model on the training set
        a = CorelsClassifier(c=0.8, n_iter=100000).fit(new_x, lof_predictions).score(new_x, lof_predictions)







