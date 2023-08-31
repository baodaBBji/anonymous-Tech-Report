import pdb

import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

from load_data import load_data
from main import calculate, traverse

sns.set_theme(style='darkgrid')

class Equal_Size_Kmeans():
    def __init__(self, repeat_times=3):
        self.repeat_times = repeat_times
        self.clusters = []
        self.kmeans = KMeans(n_clusters=2)
        if repeat_times > 1:
            for i in range(2):
                self.clusters.append(Equal_Size_Kmeans(repeat_times=repeat_times-1))

    def fit(self, X):
        self.labels_ = np.zeros(len(X))
        # Fit data with KMeans
        self.kmeans.fit(X)

        # Adjust the labels to balance them
        self.labels_ = self.kmeans.labels_
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        if counts[0] < counts[1]:
            self.small_group = 0
            idxes, self.delta = self.label_to_flip(self.kmeans.cluster_centers_[unique_labels[0]],
                                       self.kmeans.cluster_centers_[unique_labels[1]],
                                       X[np.where(self.labels_ == unique_labels[1])], counts[1] - len(X) // 2)
            idxes = np.where(self.labels_==unique_labels[1])[0][idxes]
            self.labels_[idxes] = unique_labels[0]
        elif counts[0] > counts[1]:
            self.small_group = 1
            idxes, self.delta = self.label_to_flip(self.kmeans.cluster_centers_[unique_labels[1]],
                                       self.kmeans.cluster_centers_[unique_labels[0]],
                                       X[np.where(self.labels_ == unique_labels[0])], counts[0] - len(X) // 2)
            self.labels_[np.where(self.labels_ == unique_labels[0])[0][idxes]] = unique_labels[1]


        # Continue splitting
        if self.repeat_times > 1:
            for i in [1, 0]:
                idxes = np.where(self.labels_ == i)
                self.clusters[i].fit(X[idxes])
                self.labels_[idxes] = self.clusters[i].labels_ + i * 2 ** (self.repeat_times - 1)

        return self


    def predict(self, X):
        labels = self.kmeans.predict(X)

        if self.repeat_times > 1:
            distances0 = np.sum((X - self.kmeans.cluster_centers_[0].reshape(1, -1)) ** 2, axis=1)
            distances1 = np.sum((X - self.kmeans.cluster_centers_[1].reshape(1, -1)) ** 2, axis=1)
            deltas = distances0 - distances1

            if self.small_group == 0:
                # if small_group is 0, then we should let idxes0 become larger
                idxes0 = np.where(deltas <= self.delta)[0]
                idxes1 = np.where(deltas > self.delta)[0]
            else:
                idxes0 = np.where(deltas <= - self.delta)[0]
                idxes1 = np.where(deltas > - self.delta)[0]

            for i in [1, 0]:
                idxes = eval(f"idxes{i}")
                labels[idxes] = self.clusters[i].predict(X[idxes]) + i * 2 ** (self.repeat_times - 1)

        # if self.repeat_times > 1:
        #     for i in [1, 0]:
        #         idxes = np.where(labels == i)
        #         labels[idxes] = self.clusters[i].predict(X[idxes]) + i * 2 ** (self.repeat_times-1)
        return labels



    def label_to_flip(self, center0, center1, X, num):
        '''
        params:
        center0: the center of the cluster that has more points
        center1: the center of the cluster that has fewer points
        X: the data in cluster 0
        num: the number of points that should be transferred from cluster0 to cluster1

        return:
        the idxes of X that should be flipped
        '''
        distances0 = np.sum((X - center0.reshape(1, -1)) ** 2, axis=1)
        distances1 = np.sum((X - center1.reshape(1, -1)) ** 2, axis=1)
        deltas = distances0 - distances1
        idxes = np.argsort(deltas)[:num]
        delta = deltas[idxes[-1]]
        return idxes, delta


def draw_cluster(cluster, color):
    sns.scatterplot(cluster[:, 0], cluster[:, 1], c=[color] * len(cluster))


def draw_classifier(classifier, color):
    a, b = classifier.coef_
    c = classifier.intercept_

    if b != 0:
        # a*x + b*y + c = 0 ==> y = (-ax-c)/b
        x = np.linspace(-8, 8, 2000)
        y = - (a * x + c) / b
    else:
        if a == 0:
            print("a = 0, b = 0")
            return
        else:
            x = np.array([-c / a] * 150)
            y = np.linspace(-8, 8, 150)

    if np.max(abs(y)) > 8:
        idxes = np.where((y < 8) & (y > -8))
        y = y[idxes]
        x = x[idxes]

    plt.plot(x, y, color=color)


class LocalDT():
    def __init__(self, num, device='cpu'):
        self.num = num
        self.decision_trees = None
        self.clustesr = None
        self.centers = None
        self.predictions = None
        self.device = device

    def update_decision_tree(self, X, y, threshold, multi_class):
        self.decision_trees = []
        for idxes in self.clusters:
            self.decision_trees.append(
            calculate(X[idxes], y[idxes], max_length=max_length, threshold=threshold, print_result=False,
                          multi_class=multi_class))

    def update_clusters(self, X, y):

        idx2clusters = np.zeros(len(X))
        for i in range(len(self.clusters)):
            idx2clusters[self.clusters[i]] = i

        new_clusters = []
        for i in range(len(self.clusters)):
            new_clusters.append([])

        all_preds = []
        for i, dt in enumerate(self.decision_trees):
            all_preds.append(dt.predict(X))
        for idx in range(len(X)):
            classify_correct = np.zeros(len(self.clusters))

            for i, dt in enumerate(self.decision_trees):
                if all_preds[i][idx] == y[idx]:
                    classify_correct[i] += 1

            distances = np.zeros(len(self.clusters))
            for i, c in enumerate(self.centers):
                distances[i] = np.sqrt(np.sum((X[idx] - c) ** 2) / X.shape[1])

            cluster_score = 1 * distances - classify_correct

            new_clusters[np.argmin(cluster_score)].append(idx)

        self.clusters = []
        new_decision_trees = []
        for cluster, dt in zip(new_clusters, self.decision_trees):
            if len(cluster) > 0:
                new_decision_trees.append(dt)
                self.clusters.append(cluster)
        self.centers = [np.mean(X[idxes], axis=0) for idxes in self.clusters]
        self.decision_trees = new_decision_trees

    def calculate_loss_acc(self, X, y):
        predictions = []
        y_true = []
        f1_scores = []
        for i, idxes in enumerate(self.clusters):
            predictions.append(self.decision_trees[i].predict(X[idxes]))
            y_true.append(y[idxes])
            if len(np.unique(predictions[-1])) == 1 and len(np.unique(y[idxes])) == 1:
                f1_scores.append(1 if np.unique(predictions[-1])[0] == np.unique(y[idxes])[0] else 0)
            else:
                if len(np.unique(y)) > 2:
                    f1_scores.append(metrics.accuracy_score(y[idxes], predictions[-1]))
                else:
                    f1_scores.append(metrics.f1_score(y[idxes], predictions[-1]))
        y_true = np.concatenate(y_true)
        predictions = np.concatenate(predictions)

        if len(np.unique(y)) > 2:
            return np.array(f1_scores), metrics.accuracy_score(y_true, predictions)
        else:
            return np.array(f1_scores), metrics.f1_score(y_true, predictions)


    def initialize_clusters(self, X, n_clusters, random_state=0):
        # assert n_clusters == 2
        kmeans = Equal_Size_Kmeans(repeat_times=int(math.log(n_clusters, 2))).fit(X)
        clusters = []
        clusters_idxes = []
        for _ in range(n_clusters):
            clusters.append([])
            clusters_idxes.append([])
        for idx, (x, label) in enumerate((zip(X, kmeans.labels_))):
            clusters[label].append(idx)
        return clusters, kmeans

    def delete(self, clusters, idxes):
        new_clusters = []
        for i in range(len(clusters)):
            if i not in idxes:
                new_clusters.append(clusters[i])
        return new_clusters

    def run(self, X, y, threshold):
        # initialize

        multi_class = len(np.unique(y)) > 2
        # clusters = [np.concatenate(c) for c in clusters]
        clusters, _ = self.initialize_clusters(X, self.num)
        self.clusters = clusters
        self.centers = [np.mean(X[idxes], axis=0) for idxes in self.clusters]

        print([len(c) for c in self.clusters])

        # run the process via iteration
        while True:
            i = len(self.clusters)
            self.update_decision_tree(X, y, threshold=(threshold - 0.2) + 0.2 / 9 * (i-2), multi_class=multi_class)

            f1_scores, f1_score = self.calculate_loss_acc(X, y)
            print(f"epoch: {i}, F1 score: {f1_score}")
            sys.stdout.flush()
            print(f1_scores)

            if f1_score > threshold or i > 10:
                break

            clusters_to_delete = []

            if (f1_scores < 0.9).any():
                idx = np.argmin(f1_scores)
                if len(self.clusters[idx]) <= self.num:
                    continue
                clusters, kmeans = self.initialize_clusters(X[self.clusters[idx]], 2)

                clusters = [np.array(self.clusters[idx])[tmp_idxes].tolist() for tmp_idxes in clusters]

                centers = [np.mean(X[idxes], axis=0) for idxes in clusters]
                self.clusters.extend(clusters)
                clusters_to_delete.append(idx)
                self.centers.extend(centers)

            self.clusters = self.delete(self.clusters, clusters_to_delete)
            self.centers = self.delete(self.centers, clusters_to_delete)

        print("==" * 10)

        num = 0
        length = 0

        all_rule_nums = []
        all_rule_lengths = []
        all_cluster_lengths = []

        for i in range(len(self.clusters)):
            rules = []
            traverse(self.decision_trees[i], rules)
            all_rule_lengths.append(np.sum([len(r.attributes) for r in rules]))
            all_rule_nums.append(len(rules))
            num += len(rules)
            length += np.sum([len(r.attributes) for r in rules])

            all_cluster_lengths.append(len(self.clusters[i]))


        print(pd.DataFrame({
            "rule_number": all_rule_nums,
            "rule_length": all_rule_lengths,
            "cluster_number": all_cluster_lengths
        }))

        print(f"In all, the number is: {num}, length is: {length}")
        return num, length, len(self.clusters)


    def draw(self, X):
        fig = plt.figure(figsize=(7, 6))
        fig.tight_layout()
        colors = np.array(['#377eb8', '#ff7f00', '#0000FF', '#FFFF00',
                           '#FFA500', '#FF0000', '#008000', '#808080',
                           '#800080', '#FFD700', "#B1BBB8", "#F3E6C3"])
        np.random.shuffle(colors)

        for i in range(len(self.clusters)):
            draw_cluster(X[self.clusters[i]], color=colors[i])
        plt.savefig(f"./figures/{name}_cluster.pdf", format='pdf', dpi=600)
        plt.show()


def main(name, num, data_dir, threshold):
    print("Name:", name)

    def run_lof(X, y, num_outliers=560, k=60):
        clf = LocalOutlierFactor(n_neighbors=k)
        clf.fit(X)
        lof_scores = -clf.negative_outlier_factor_
        threshold = np.sort(lof_scores)[::-1][num_outliers]
        lof_predictions = np.array(lof_scores > threshold)
        lof_predictions = np.array([int(i) for i in lof_predictions])
        print("F-1 score:", metrics.f1_score(y, lof_predictions))
        return lof_predictions, lof_scores


    X, y = load_data(data_dir=data_dir)
    lof_predictions = lof_scores = y

    local_lime = LocalDT(num=num)

    number, length, n_c = local_lime.run(X, lof_predictions, threshold)
    return number, length, n_c, local_lime.clusters


if __name__ == '__main__':


    datasets = ['Wine']

    numbers = [2, 4, 8]
    # numbers = [2]
    # thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # thresholds = [0.90]
    thresholds = [0.8]
    # max_lengths = [2, 4, 6, 8, 10, 12]
    max_lengths = [10]
    data_dir = "../data/"
    # data_dir = '/data/wangyu/data/'

    results = []

    for name in datasets:
        best_length = 1e9
        for num in numbers:
            for threshold in thresholds:
                for max_length in max_lengths:
                    number, length, n_c, clusters = main(name, num=num, data_dir=data_dir, threshold=threshold)
                    results.append([length, number, n_c])
                    if length < best_length and threshold == 0.8:
                        np.save(f"./L-STAIR/{name}_{threshold}_{num}.npy", np.array(clusters))
                        best_length = length


        print(results)
