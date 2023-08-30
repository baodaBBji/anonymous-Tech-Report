import numpy as np
from load_data import load_data
from sklearn import metrics
import smote_variants as sv
from decision_tree import run_decision_tree
from sklearn.neighbors import LocalOutlierFactor
import sys


if len(sys.argv) == 1:
    name = 'PageBlock'
else:
    name = sys.argv[1]

X0, y0, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range = load_data(name)
X0, y0 = np.array(X0), np.array(y0)


oversampler = sv.distance_SMOTE()

criterion = 'gini'

def run_lof(X, y, num_outliers=560, k=60):
    clf = LocalOutlierFactor(n_neighbors=k)
    clf.fit(X)
    lof_scores = -clf.negative_outlier_factor_
    threshold = np.sort(lof_scores)[::-1][num_outliers]
    lof_predictions = np.array(lof_scores > threshold)
    lof_predictions = np.array([int(i) for i in lof_predictions])
    print("F-1 score:", metrics.f1_score(y, lof_predictions))
    return lof_predictions, lof_scores



lof_predictions, lof_scores = run_lof(X0, y0, k=lof_krange[7], num_outliers=int(np.sum(y0)))


neg_idxes = np.where(lof_predictions == 0)[0]
pos_idxes = np.where(lof_predictions == 1)[0]
pos_X = X0[pos_idxes]
neg_X = X0[neg_idxes]

X = X0.copy().tolist()
y = y0.copy().tolist()
lof_predictions = lof_predictions.tolist()

scale = int(len(neg_X) / len(pos_X))
for _ in range(1 - 1):
    for pos_idx, temp_X in zip(pos_idxes, pos_X):
        X.append(temp_X)
        lof_predictions.append(lof_predictions[pos_idx])
        y.append(y0[pos_idx])

X = np.array(X)
lof_predictions = np.array(lof_predictions)
y = np.array(y)
print("before oversample, X:", X0.shape)
print("after oversample, X:", X.shape)

run_decision_tree(X, y, lof_predictions, lof_scores, criterion)
