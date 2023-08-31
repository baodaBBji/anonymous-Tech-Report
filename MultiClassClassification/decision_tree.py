import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def run_decision_tree(X, y, lof_predictions, lof_scores, criterion):

    OUTLIER_PERCENTAGE = int(np.sum(y)) / len(y)

    def run_decisiontreeclassifier(X, preds):
        clf = DecisionTreeClassifier(criterion=criterion, random_state=0, min_samples_leaf=4, max_depth=6,
                                     class_weight={0: np.sum(y) / len(y), 1: 1.0})
        clf.fit(X, preds)

        leaf_nodes = clf.apply(X)
        return leaf_nodes

    from sklearn.tree import DecisionTreeClassifier
    leaf_nodes = run_decisiontreeclassifier(X, lof_predictions)
    print('total leaf node size: ', len(set(leaf_nodes)))

    for leaf in set(leaf_nodes):
        total_examples_ids = np.where(leaf_nodes == leaf)[0]
        total_num_examples = sum(leaf_nodes == leaf)
        temp_num_inliers = total_examples_ids[np.where(y[leaf_nodes == leaf] == 0)[0]]
        temp_num_outliers = total_examples_ids[np.where(y[leaf_nodes == leaf] == 1)[0]]
        print(leaf, len(temp_num_inliers), len(temp_num_outliers))


    labeled_outlier_ids = []
    labeled_inlier_ids = []
    num_labeled_rules = 0

    need_to_process = dict()
    for leaf in set(leaf_nodes):
        total_examples_ids = np.where(leaf_nodes == leaf)[0]
        total_num_examples = sum(leaf_nodes == leaf)
        temp_num_inliers = total_examples_ids[np.where(y[leaf_nodes == leaf] == 0)[0]]
        temp_num_outliers = total_examples_ids[np.where(y[leaf_nodes == leaf] == 1)[0]]
        if (len(temp_num_outliers) * 1.0 / total_num_examples >= 0.9
                or len(temp_num_inliers) * 1.0 / total_num_examples >= 0.9):
            num_labeled_rules = num_labeled_rules + 1
            if (len(temp_num_outliers) > len(temp_num_inliers)):
                labeled_outlier_ids.extend(temp_num_inliers)
                labeled_outlier_ids.extend(temp_num_outliers)
            else:
                labeled_inlier_ids.extend(temp_num_inliers)
                labeled_inlier_ids.extend(temp_num_outliers)
        elif total_num_examples <= 5:
            num_labeled_rules += total_num_examples
            labeled_inlier_ids.extend(temp_num_inliers)
            labeled_outlier_ids.extend(temp_num_outliers)
        else:
            need_to_process[leaf] = total_examples_ids

    depth = 1
    while len(need_to_process) > 0 and depth < 2:
        temp_need_to_process = dict()
        for leaf_id in need_to_process:
            total_examples_ids = need_to_process[leaf_id]
            partX = X[total_examples_ids]
            partY = y[total_examples_ids]
            if (sum(partY) * 1.0 / len(partY) >= max(OUTLIER_PERCENTAGE, 0.1)):
                num_outliers = int(max(len(partY) / 2, int(sum(partY))))  # int(sum(partY)) #
            else:
                num_outliers = int(max(int(sum(partY)), int(len(partY) * OUTLIER_PERCENTAGE)))
            threshold = np.sort(lof_scores[total_examples_ids])[::-1][num_outliers]
            part_lof_predictions = np.array(lof_scores[total_examples_ids] > threshold)
            part_lof_predictions = np.array([int(i) for i in part_lof_predictions])

            leaf_nodes_part = run_decisiontreeclassifier(partX, part_lof_predictions)

            for leaf2 in set(leaf_nodes_part):
                total_examples_ids_part = total_examples_ids[np.where(leaf_nodes_part == leaf2)[0]]
                total_num_examples = sum(leaf_nodes_part == leaf2)
                temp_num_inliers = total_examples_ids_part[np.where(partY[leaf_nodes_part == leaf2] == 0)[0]]
                temp_num_outliers = total_examples_ids_part[np.where(partY[leaf_nodes_part == leaf2] == 1)[0]]
                if (len(temp_num_outliers) * 1.0 / total_num_examples >= 0.9
                        or len(temp_num_inliers) * 1.0 / total_num_examples >= 0.9):
                    num_labeled_rules = num_labeled_rules + 1
                    if (len(temp_num_outliers) > len(temp_num_inliers)):
                        labeled_outlier_ids.extend(temp_num_inliers)
                        labeled_outlier_ids.extend(temp_num_outliers)
                    else:
                        labeled_inlier_ids.extend(temp_num_inliers)
                        labeled_inlier_ids.extend(temp_num_outliers)
                elif total_num_examples <= 5:
                    num_labeled_rules += total_num_examples
                    labeled_inlier_ids.extend(temp_num_inliers)
                    labeled_outlier_ids.extend(temp_num_outliers)
                else:
                    temp_need_to_process[leaf2] = total_examples_ids_part
        need_to_process = temp_need_to_process
        depth = depth + 1

    print("Total Number of Labeled Outliers: ", len(labeled_outlier_ids))
    print("Total Number of Labeled Inliers: ", len(labeled_inlier_ids))
    print("Total Number of rules: ", num_labeled_rules)

    all_label_index = np.union1d(labeled_outlier_ids, labeled_inlier_ids)
    predicted_labels = lof_predictions.copy()
    predicted_labels[labeled_outlier_ids] = 1
    predicted_labels[labeled_inlier_ids] = 0

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X[all_label_index.astype(int)], predicted_labels[all_label_index.astype(int)])
    y_pred0 = clf.predict(X)
    print("F-1 score:", metrics.f1_score(y, y_pred0))
    print("Accuracy score: ", metrics.accuracy_score(y, y_pred0))

    # mislabeled outliers, the missing outliers
    idxes = np.where(y_pred0 == 0)
    missing_outliers_num = len(np.where(y[idxes] == 1)[0])
    print("missing_outliers_num:", missing_outliers_num)
    print("missing outliers ratio:", missing_outliers_num / np.sum(y))
