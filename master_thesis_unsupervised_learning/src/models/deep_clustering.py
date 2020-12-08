""" script for building and training a deep embedded clustering model
  using a pre-trained conv autoencoder """

from pathlib import Path
import json
import numpy as np
from datetime import datetime
import mlflow
from tqdm import tqdm_notebook as tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import os

from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Layer, InputSpec

from src.models.base_model import BaseModel
from src.models.conv_autoencoder import ConvAutoEncoder
from src.data.preprocessing import Preprocessing
from src.models.metrics import match_labels_and_clusters, acc, nmi, ari

# Setting seeds for reproducibility
SEED = 123

# Labels used for normal and anomalous samples
NORMAL = 0
ANOMALY = 1

PROJECT_ROOT = Path(__file__).parent.parent.parent


class DeepClustering(BaseModel):

    def __init__(self, load_from, maxiter=800, update_interval=8,
                 tol=0.001, batch_size=128, nb_clusters=12,
                 highest_io_class=4, verbose=1, **kwargs):
        """
        define and build a deep clustering model
        :param load_from: directory path where the pre-trained
        conv autoencoder and its properties are saved
        :param maxiter: the maximum number of iteration while
        training the deep clustering model
        :param update_interval: update the target distribution
        after "update_interval" iterations
        :param tol: tolerance to stop training
        :param batch_size: batch size used in the method fit,
        can be updated in the method custom_fit
        :param nb_clusters: number of clusters, can be updated
        in method custom_fit
        :param highest_io_class: the highest label of io classes
        default 4 because we have 5 io groups in the default dataset
        :param verbose: if greater than 0 show info messages
        """
        super().__init__()
        self.load_from = load_from
        # load pre-trained conv autoencoder
        self.convae = ConvAutoEncoder.load(self.load_from)
        self.maxiter = maxiter
        self.update_interval = update_interval
        self.tol = tol
        self.batch_size = batch_size
        self.nb_clusters = nb_clusters
        self.map_cluster_to_labels = {}
        self.io_list = []
        self.nio_list = []
        self.highest_io_class = highest_io_class
        if kwargs:
            # if already trained load mapping from json files
            for key, value in kwargs.items():
                self.map_cluster_to_labels[key] = int(value)
            for v in self.map_cluster_to_labels.values():
                if 0 <= v <= self.highest_io_class:
                    self.io_list.append(v)
                elif v > self.highest_io_class:
                    self.nio_list.append(v)
        self.verbose = verbose
        self.deep_model, self.encoder = self.build()
        self.compile(opt="adam", loss=["kld", "binary_crossentropy"],
                     loss_weights=[0.04, 1])

    def build(self):
        """
        method used to build the deep clustering and encoder models
        :return: deep clustering and encoder models
        """
        clustering_layer = ClusteringLayer(n_clusters=self.nb_clusters,
                                           name='clustering')(self.convae.encoder.output)
        deep_model = Model(inputs=self.convae.model.input,
                           outputs=[clustering_layer, self.convae.model.output])
        encoder = Model(inputs=deep_model.input,
                        outputs=deep_model.get_layer('latent_space').output)
        return deep_model, encoder

    @staticmethod
    def target_distribution(q):
        """
        computes an auxiliary target distribution
        :param q: actual distribution computed with the deep
        clustering model
        :return: target distribution
        """
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, opt, loss, loss_weights):
        self.deep_model.compile(optimizer=opt,
                                loss=loss,
                                loss_weights=loss_weights)

    def __plot_embedded_space(self, train, target):
        """
        method used to plot the embedded space of the model
        after applying t-sne
        :param train: input data
        :param target: labels
        """
        x_encoded = self.encoder.predict(train)
        X_2d = self.__apply_tsne(x_encoded)

        fig1, ax1 = plt.subplots()
        fig1.set_size_inches((16, 10))
        scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=target, s=40, cmap=plt.cm.Paired)
        legend1 = ax1.legend(*scatter.legend_elements(),
                             loc="upper right", title="labels")
        ax1.add_artist(legend1)
        plt.show()

    def __get_scatter_plot(self, data, labels, legend_name):
        """
        method used to return the embedded space scatter plot
        of the model
        :param data: 2d input data
        :param labels: corresponding labels
        :param legend_name: legend name
        :return: scatter plot figure
        """
        fig, ax = plt.subplots()
        fig.set_size_inches((16, 10))
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels,
                             s=40, cmap=plt.cm.Paired)
        legend = ax.legend(*scatter.legend_elements(),
                           loc="upper right", title=legend_name)
        ax.add_artist(legend)
        return fig

    def __apply_tsne(self, data):
        """
        method used to apply the t-sne algorithm to reduce
        the problem dimension to 2d
        :param data: input data
        :return: 2d data
        """
        restore_these_settings = np.geterr()

        temp_settings = restore_these_settings.copy()
        temp_settings["over"] = "ignore"
        temp_settings["under"] = "ignore"

        np.seterr(**temp_settings)
        tsne2 = TSNE(n_components=2, random_state=SEED)
        X_2d = tsne2.fit_transform(data)
        np.seterr(**restore_these_settings)
        return X_2d

    def custom_fit(self, train, target, mlflow_experiment, representation,
                   maxiter=800, update_interval=8, batch_size=128, tol=0.001,
                   test=None, target_test=None, viz=True):
        """
        custom fit method that trains the deep clustering
        models with the given parameters
        :param train: input dataset used for training
        :param target: corresponding labels used for evaluating the model
        :param mlflow_experiment: name of the mlflow context used for training
        :param representation: experiment name used in mlflow e.g.
        "melspectrogram" or "mfcc" etc..
        :param maxiter: the maximum number of iteration while
        training the deep clustering model
        :param update_interval: update the target distribution
        after "update_interval" iterations
        :param batch_size: batch size used for training
        :param tol: tolerance to stop training
        :param test: test dataset used for testing the performance of the model
        :param target_test: corresponding labels of the test dataset
        :param viz: either to plot reduced embedded space during training
        :return: Keras history after using fit
        """
        self.maxiter = maxiter
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.tol = tol
        init_f1_score_test = 0.0

        start_time = datetime.now()  # save start time to compute training duration
        # set mlflow experiment name
        mlflow.set_experiment(mlflow_experiment)
        if self.verbose > 0:
            print("[INFO] mlflow experiment: {}".format(mlflow_experiment))
            print("[INFO] deep clustering training starting ...")

        loss = 0
        index = 0
        index_array = np.arange(train.shape[0])
        # init clustering layer weights
        y_pred = self.convae.kmeans.predict(self.convae.encoder.predict(train))
        self.deep_model.get_layer(name='clustering').set_weights([self.convae.kmeans.cluster_centers_])
        y_pred_last = np.copy(y_pred)

        with mlflow.start_run():  # run within mlflow context

            for ite in tqdm(range(maxiter)):
                if ite % update_interval == 0:
                    q, _ = self.deep_model.predict(train, verbose=self.verbose)
                    p = self.target_distribution(q)  # update the auxiliary target distribution p
                    # evaluate the clustering performance
                    y_pred = q.argmax(1)

                    if viz:
                        print("[INFO] iteration : {}".format(ite))
                        if target is not None:
                            self.__plot_embedded_space(train, target)
                        else:
                            self.__plot_embedded_space(train, y_pred)

                    if target is not None:
                        acc_metric = np.round(acc(target, y_pred), 5)
                        nmi_metric = np.round(nmi(target, y_pred), 5)
                        ari_metric = np.round(ari(target, y_pred), 5)
                        new_y_pred, map_cluster_to_labels = \
                            match_labels_and_clusters(target, y_pred)
                        # update mapping clusters labels
                        self.set_mapping_cluster_labels(map_cluster_to_labels,
                                                        self.highest_io_class)

                        loss = np.round(loss, 5)
                        print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' %
                              (ite, acc_metric, nmi_metric, ari_metric), ' ; loss=', loss)

                        if test is not None and target_test is not None:
                            q_test, _ = self.deep_model.predict(test)
                            y_pred_test = q_test.argmax(1)
                            new_y_pred_test = np.array([self.map_cluster_to_labels[str(elem)]
                                                        for elem in y_pred_test])
                            f1_score_test = np.round(f1_score(target_test, new_y_pred_test,
                                                              average='macro'), 3)
                            acc_test = np.round(acc(target_test, y_pred_test), 3)
                            # save initial f1_score_test to see the improvement
                            if ite == 0:
                                init_f1_score_test = f1_score_test
                            if self.verbose > 0:
                                print("[INFO] testset f1-score : ", f1_score_test)
                                print("[INFO] testset acc: ", acc_test)

                    # check stop criterion
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    if ite > 0 and delta_label < self.tol:
                        print('delta_label ', delta_label, '< tol ', tol)
                        print('Reached tolerance threshold. Stopping training.')

                        # log parameters to mlflow
                        mlflow.log_params({
                            "1.representation": representation,
                            "2.init_f1_score_t": init_f1_score_test,
                            "3.nb_clusters": self.nb_clusters,
                            "4.batch_size": self.batch_size,
                            "5.iteration": ite,
                            "6.update_interval": self.update_interval
                        })

                        # log metrics to mlflow
                        mlflow.log_metrics({
                            "1.f1_score": np.round(f1_score(target, new_y_pred, average='macro'), 3),
                            "2.recall": np.round(recall_score(target, new_y_pred, average='macro'), 3),
                            "3.precision": np.round(precision_score(target, new_y_pred, average='macro'), 3),
                            "4.acc": np.round(acc(target, new_y_pred), 3),
                            "5.nmi": np.round(nmi(target, new_y_pred), 3),
                            "6.ari": np.round(ari(target, new_y_pred), 3),
                            "9.loss_cls": loss[0],
                            "9.loss_rec": loss[1]
                        })
                        if test is not None and target_test is not None:
                            mlflow.log_metrics({
                                "7.f1_score_t": f1_score_test,
                                "8.acc_t": acc_test
                            })

                        x_encoded = self.encoder.predict(train)
                        # save scatter plot of latent space after applying t-sne
                        X_2d = self.__apply_tsne(x_encoded)
                        fig1 = self.__get_scatter_plot(data=X_2d, labels=target,
                                                       legend_name='labels')
                        fig1.savefig("scatter.png")

                        # save scatter plot of kmeans clusters
                        fig2 = self.__get_scatter_plot(data=X_2d, labels=new_y_pred,
                                                       legend_name='clusters')
                        fig2.savefig("kmeans_clusters.png")

                        # log artifacts to mlflow
                        mlflow.log_artifact("scatter.png")
                        mlflow.log_artifact("kmeans_clusters.png")
                        # remove artifacts after logging
                        os.remove("scatter.png")
                        os.remove("kmeans_clusters.png")
                        break
                idx = index_array[index * self.batch_size: min((index + 1) * self.batch_size,
                                                               train.shape[0])]
                loss = self.deep_model.train_on_batch(x=train[idx], y=[p[idx], train[idx]])
                index = index + 1 if (index + 1) * batch_size <= train.shape[0] else 0
        end_time = datetime.now()
        if self.verbose > 0:
            print('[INFO] deep clustering training completed ... duration: {}'
                  .format(end_time - start_time))

    def set_mapping_cluster_labels(self, map_cluster_to_labels, highest_io_class):
        """
        method for setting a mapping between predicted clusters and
        the true labels, can be set after training automatically as well as
        manually by user afterwards
        :param map_cluster_to_labels: dictionary containing the mapping
        between clusters and labels e.g. {"0": 1, "1": 0} - io labels are
        between 0 and (nb_io_classes - 1), nio labels are greater than or equal
        to nb_io_classes
        :param highest_io_class: the highest number of io classes, default 4
        for the default dataset, if the dataset is only split into 2 categories
        anomlies and normal sample, then highest_io_class must be set to 0
        """
        self.highest_io_class = highest_io_class
        self.map_cluster_to_labels = map_cluster_to_labels
        for v in self.map_cluster_to_labels.values():
            if 0 <= v <= self.highest_io_class:
                self.io_list.append(v)
            elif v >= self.highest_io_class:
                self.nio_list.append(v)

    @classmethod
    def load(cls, model_path: str):
        load_path = Path(PROJECT_ROOT / model_path)
        with open(Path(load_path / "properties.json"), 'r') as fp:
            prop_dict = json.load(fp)
            deep_clustering = cls(**prop_dict)
        deep_clustering.deep_model = load_model(Path(load_path / "deep_clustering.h5"), custom_objects={'ClusteringLayer': ClusteringLayer})
        deep_clustering.encoder = Model(inputs=deep_clustering.deep_model.input,
                                        outputs=deep_clustering.deep_model.get_layer('latent_space').output)
        return deep_clustering

    def save(self, model_path: str):
        save_path = Path(PROJECT_ROOT / model_path)
        save_path.mkdir(exist_ok=True)
        self.deep_model.save(Path(save_path / "deep_clustering.h5"))
        prop_dict = {
            "load_from": self.load_from,
            "maxiter": int(self.maxiter),
            "update_interval": int(self.update_interval),
            "tol": self.tol,
            "batch_size": int(self.batch_size),
            "nb_clusters": int(self.nb_clusters),
            "verbose": int(self.verbose)
        }

        if self.map_cluster_to_labels:
            # add mapping between clusters and labels for prediction
            for key, value in self.map_cluster_to_labels.items():
                prop_dict[key] = int(value)
        with open(Path(save_path / "properties.json"), 'w') as fp:
            json.dump(prop_dict, fp)

    def predict(self, samples):

        # classify samples using deep clustering model
        q, _ = self.deep_model.predict(samples)
        predicted = q.argmax(1)
        if self.map_cluster_to_labels:
            predicted = np.array([self.map_cluster_to_labels[str(elem)]
                                  for elem in predicted])
        return predicted

    def custom_metrics(self, data, target):

        if not self.map_cluster_to_labels:
            print("please use set_mapping_cluster_labels() to define the"
                  " mapping between clusters and labels as dictionary - "
                  " cluster number as string (key) -> true label number (value)"
                  " as integer.")

        # classify samples using deep clustering model
        q, _ = self.deep_model.predict(data)
        predicted = q.argmax(1)
        new_predicted = np.array([self.map_cluster_to_labels[str(elem)]
                                  for elem in predicted])

        precision = np.round(precision_score(target, new_predicted, average='macro'),
                             decimals=3)
        recall = np.round(recall_score(target, new_predicted, average='macro'),
                          decimals=3)
        score = np.round(f1_score(target, new_predicted, average='macro'),
                         decimals=3)
        if self.verbose > 0:
            print("[INFO] score: ", score)
            print("[INFO] precision: ", precision)
            print("[INFO] recall: ", recall)
        return precision, recall, score

    def fit(self, x_train: list, y_train: list = None):
        train, _ = Preprocessing.dataset_from_files(io_files=x_train,
                                                    nio_files=y_train,
                                                    normal_label=NORMAL,
                                                    anomaly_label=ANOMALY,
                                                    img_size=(128, 128),
                                                    use_color=True,
                                                    reduce_noise=True)
        start_time = datetime.now()  # save start time to compute training duration
        if self.verbose > 0:
            print("[INFO] deep clustering training starting ...")

        index = 0
        index_array = np.arange(train.shape[0])
        # init clustering layer weights
        y_pred = self.convae.kmeans.predict(self.convae.encoder.predict(train))
        self.deep_model.get_layer(name='clustering').set_weights([self.convae.kmeans.cluster_centers_])
        y_pred_last = np.copy(y_pred)

        for ite in tqdm(range(self.maxiter)):
            if ite % self.update_interval == 0:
                q, _ = self.deep_model.predict(train, verbose=self.verbose)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                # evaluate the clustering performance
                y_pred = q.argmax(1)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < self.tol:
                    print('delta_label ', delta_label, '< tol ', self.tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            idx = index_array[index * self.batch_size: min((index + 1) * self.batch_size,
                                                           train.shape[0])]
            self.deep_model.train_on_batch(x=train[idx], y=[p[idx], train[idx]])
            index = index + 1 if (index + 1) * self.batch_size <= train.shape[0] else 0
        end_time = datetime.now()
        if self.verbose > 0:
            print('[INFO] deep clustering training completed ... duration: {}'
                  .format(end_time - start_time))

    def metrics(self, test_io: list, test_nio: list):

        if not self.map_cluster_to_labels:
            print("please use set_mapping_cluster_labels() to define the"
                  " mapping between clusters and labels as dictionary - "
                  " cluster number as string (key) -> true label number (value)"
                  " as integer - labels: io must be in [0, 9] and nio in [10, 19].")
            return

        test_data, test_target = Preprocessing.dataset_from_files(io_files=test_io,
                                                                  nio_files=test_nio,
                                                                  normal_label=NORMAL,
                                                                  anomaly_label=ANOMALY,
                                                                  img_size=(128, 128),
                                                                  use_color=True,
                                                                  reduce_noise=True)
        # classify samples using deep clustering model
        q, _ = self.deep_model.predict(test_data)
        predicted = q.argmax(1)
        new_predicted = np.array([self.map_cluster_to_labels[str(elem)]
                                  for elem in predicted])
        # build two clusters io and nio
        predicted_two_clusters = np.array([NORMAL if elem in self.io_list
                                           else ANOMALY for elem in new_predicted])
        precision = np.round(precision_score(test_target, predicted_two_clusters),
                             decimals=3)
        recall = np.round(recall_score(test_target, predicted_two_clusters),
                          decimals=3)
        score = np.round(f1_score(test_target, predicted_two_clusters),
                         decimals=3)
        return precision, recall, score

    def predict_files(self, file_paths: list):

        if not self.map_cluster_to_labels:
            print("please use set_mapping_cluster_labels() to define the"
                  " mapping between clusters and labels as dictionary - "
                  " cluster number as string (key) -> true label number (value)"
                  " as integer - labels: io must be in [0, 9] and nio in [10, 19].")
            return

        data, _ = Preprocessing.dataset_from_files(io_files=file_paths,
                                                   img_size=(128, 128),
                                                   use_color=True,
                                                   reduce_noise=True)

        predicted = self.predict(data)
        # build two clusters io and nio
        predicted_two_clusters = np.array([NORMAL if elem in self.io_list
                                           else ANOMALY for elem in predicted])
        # numpy array contains True if prediction is an anomaly,
        # otherwise False
        is_anomaly = np.array([prediction == ANOMALY for
                               prediction in predicted_two_clusters])
        return is_anomaly


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))  # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
