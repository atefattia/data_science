import json
from pathlib import Path
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent


class ClusterMerge:

    def __init__(self, dict_labels_path, similarity_metric="min"):
        """

        :param dict_labels_path:
        :param similarity_metric: min or max or average
        """
        full_path = Path(PROJECT_ROOT / dict_labels_path)

        with open(Path(full_path), 'r') as fp:
            self.semi_supervised_mapping = json.load(fp)

        self.initial_nb_clusters = None
        # number of keys in the json files
        self.category_nb = len(self.semi_supervised_mapping)
        self.similarity_metric = similarity_metric

        # define some rules with given labels -- used to
        # merge clusters, merge only if constraint is valid
        self.cannot_link = self.init_cannot_link()
        self.level_target = {}

    def init_cannot_link(self):
        init_list = []
        for k, v in self.semi_supervised_mapping.items():
            for elem in v:
                for k_aux, v_aux in self.semi_supervised_mapping.items():
                    if k == k_aux:
                        continue
                    for elem_aux in v_aux:
                        init_list.append((elem, elem_aux))
        return init_list

    def clusters_distance(self, data, actual_target):
        nb_clusters = len(np.unique(actual_target))
        distance_matrix = np.full(shape=(nb_clusters, nb_clusters), fill_value=float('inf'))
        for i in range(nb_clusters):
            cluster_i = data[actual_target == i]
            for j in range(i + 1, nb_clusters):
                cluster_j = data[actual_target == j]
                distance_matrix[i, j] = self.compute_similarity(cluster_i, cluster_j)
        return distance_matrix

    def compute_similarity(self, cluster_i, cluster_j):
        similarity = 0
        if self.similarity_metric == "min":
            min_value = float('inf')
            for clus_i in cluster_i:
                for clus_j in cluster_j:
                    if np.linalg.norm(clus_i - clus_j) < min_value:
                        min_value = np.linalg.norm(clus_i - clus_j)
            similarity = min_value
        elif self.similarity_metric == "max":
            max_value = -float('inf')
            for clus_i in cluster_i:
                for clus_j in cluster_j:
                    if np.linalg.norm(clus_i - clus_j) > max_value:
                        max_value = np.linalg.norm(clus_i - clus_j)
            similarity = max_value
        elif self.similarity_metric == "average":
            dist = cdist(cluster_i, cluster_j, metric="euclidean")
            dist_sum = np.sum(dist)
            avg_value = dist_sum / (cluster_i.shape[0] * cluster_j.shape[0])
            similarity = avg_value

        return similarity

    def constraint_valid(self, actual_target, i, j):
        cluster_i_indices = np.argwhere(actual_target == i)
        cluster_i_indices = cluster_i_indices.reshape(cluster_i_indices.shape[0])
        cluster_j_indices = np.argwhere(actual_target == j)
        cluster_j_indices = cluster_j_indices.reshape(cluster_j_indices.shape[0])
        for pair in self.cannot_link:
            if pair[0] in cluster_i_indices and pair[1] in cluster_j_indices:
                return False
            if pair[1] in cluster_i_indices and pair[0] in cluster_j_indices:
                return False
        return True

    def merge(self, new_prediction, i, j):
        if i < j:
            new_prediction[new_prediction == j] = i
        else:
            new_prediction[new_prediction == i] = j

        max_new_prediction = np.unique(new_prediction).max()
        for element in range(0, max_new_prediction):
            if element not in new_prediction:
                new_prediction[new_prediction == max_new_prediction] = element
        return new_prediction

    def merge_clusters(self, data, initial_prediction, target_nb_clusters):

        assert self.category_nb == target_nb_clusters
        new_prediction = np.copy(initial_prediction)
        self.initial_nb_clusters = len(np.unique(new_prediction))
        actual_nb_clusters = self.initial_nb_clusters
        for _ in tqdm(range(self.initial_nb_clusters - self.category_nb)):
            actual_nb_clusters = len(np.unique(new_prediction))
            distance_matrix = self.clusters_distance(data, new_prediction)
            min_dist_indices = np.argsort(distance_matrix.reshape(actual_nb_clusters ** 2))

            for k in range(len(min_dist_indices)):
                j = min_dist_indices[k] % actual_nb_clusters
                i = min_dist_indices[k] // actual_nb_clusters
                if self.constraint_valid(new_prediction, i, j):
                    new_prediction = self.merge(new_prediction, i, j)
                    break

            if actual_nb_clusters < 20:
                self.level_target[actual_nb_clusters] = new_prediction.copy()

        if actual_nb_clusters == target_nb_clusters:
            print("cluster number : ", actual_nb_clusters)
        else:
            pass  # TODO
        return new_prediction


if __name__ == '__main__':
    merge = ClusterMerge("src/models/semi_supervised_mapping", 5)
    print(merge.cannot_link)