""" script for defining metrics for unsupervised learning """

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    indices = linear_sum_assignment(w.max() - w)
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    return sum([w[i, j] for i, j in indices]) * 1.0 / y_pred.size


def match_labels_and_clusters(y_true, y_pred):
    """
    convenient method to map clusters to the corresponding true labels.
    :param y_true: contains the true labels
    :param y_pred: contains the clustering prediction

    :return:
    y_pred_new: new prediction numpy array matched with true labels
    map_cluster_to_target: mapping between clusters and true labels
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    indices = linear_sum_assignment(w.max() - w)

    indices = np.asarray(indices)
    indices = np.transpose(indices)

    map_cluster_to_target = {}
    for elem in indices:
        map_cluster_to_target[str(elem[0])] = elem[1]
    y_pred_new = np.array([map_cluster_to_target[str(i)] for i in y_pred])
    return y_pred_new, map_cluster_to_target

