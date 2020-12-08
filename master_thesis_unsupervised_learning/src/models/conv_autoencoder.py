""" script for building and training a conv autoencoder model  """

from src.models.base_model import BaseModel
from src.models.callbacks import VisualizationCallback, CustomEarlyStopping
from src.data.preprocessing import Preprocessing
from src.models.metrics import match_labels_and_clusters, acc, nmi, ari
from sklearn.model_selection import train_test_split
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import json
import mlflow
import random
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

from numpy.random import seed

import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Reshape
from keras import backend as K
from keras.models import Model
from keras.layers import MaxPool2D, UpSampling2D
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Setting seeds for reproducibility
SEED = 123

tf.random.set_seed(SEED)
random.seed(SEED)
seed(SEED)

# Labels used for normal and anomalous samples
NORMAL = 0
ANOMALY = 1

PROJECT_ROOT = Path(__file__).parent.parent.parent


class ConvAutoEncoder(BaseModel):

    def __init__(self, img_height=128, img_width=128, nb_channels=3,
                 latent_dim=5, nb_clusters=12, epochs=250, batch_size=128,
                 kernel_initializer='he_uniform', highest_io_class=4,
                 augmentation=False, verbose=1, **kwargs):
        """
        define and build a conv autoencoder
        :param img_height: input image height
        :param img_width: input image width
        :param nb_channels: number of channels of input image
        :param latent_dim: dimension of the latent space vector,
        used afterwards to make clusters
        :param nb_clusters: number of clusters used in K-Means,
        can be updated in method custom_fit
        :param epochs: number of epochs used in the method fit,
        can be updated in method custom_fit
        :param batch_size: batch size used in the method fit,
        can be updated in the method custom_fit
        :param kernel_initializer: kernel initializer of the conv layers
        :param highest_io_class: the highest label of io classes
        default 4 because we have 5 io groups in the default dataset
        :param augmentation: whether to use augmentation during the
        training
        :param verbose: if greater than 0 show info messages
        """
        super().__init__()
        self.input_shape = (img_height, img_width, nb_channels)
        self.latent_dim = latent_dim
        self.nb_clusters = nb_clusters
        self.epochs = epochs
        self.batch_size = batch_size
        # contains a kmeans model, used to make
        # predictions, updated after training
        self.kmeans = None
        self.kernel_initializer = kernel_initializer
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
        self.augmentation = augmentation
        self.verbose = verbose

        self.model, self.encoder = self.build()
        self.compile(opt="adam", loss="binary_crossentropy")

    def build(self):
        """
        method used to build the conv autoencoder and encoder models
        :return: autoencoder and encoder models
        """
        ########### ENCODER ############
        # input shape (128*128*3)
        inputs = Input(shape=self.input_shape, name='ae_input')

        # enc block 1
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                   padding='same', activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   name="conv2d_32")(inputs)  # (128*128*32)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2,
                      padding='same',
                      name="maxpool_32")(x)  # (64*64*32)

        # enc block 2
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                   padding='same', activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   name="conv2d_64")(x)  # (64*64*64)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2,
                      padding='same',
                      name="maxpool_64")(x)  # (32*32*64)

        # enc block 3
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=1,
                   padding='same', activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   name="conv2d_128")(x)  # (32*32*128)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2,
                      padding='same',
                      name="maxpool_128")(x)  # (16*16*128)

        # enc last block
        x = MaxPool2D(pool_size=(2, 2), strides=2,
                      padding='same',
                      name="maxpool_enc")(x)  # (8*8*128)

        decoder_shape = K.int_shape(x)[1:]
        # flatten
        x = Flatten(name="flatten_enc")(x)  # 8192
        x = Dense(1000)(x)
        # feature vectors
        latent_space = Dense(self.latent_dim, name="latent_space")(x)

        ########### DECODER ############
        x = Dense(1000)(latent_space)
        # dec first block
        x = Dense(np.prod(decoder_shape), activation='relu',
                  name="flatten_dec")(x)  # 8192
        x = Reshape(decoder_shape)(x)  # (8*8*128)
        x = UpSampling2D(size=(2, 2), name="maxpool_dec")(x)  # (16*16*128)

        # dec block 1
        x = Conv2DTranspose(filters=128, kernel_size=(3, 3),
                            strides=1, padding='same',
                            kernel_initializer=self.kernel_initializer,
                            activation='relu',
                            name="tconv2d_128")(x)  # (16*16*128)
        x = BatchNormalization(axis=-1)(x)
        x = UpSampling2D(size=(2, 2), name="upsamp_128")(x)  # (32*32*128)

        # dec block 2
        x = Conv2DTranspose(filters=64, kernel_size=(3, 3),
                            strides=1, padding='same',
                            kernel_initializer=self.kernel_initializer,
                            activation='relu',
                            name="tconv2d_64")(x)  # (32*32*64)
        x = BatchNormalization(axis=-1)(x)
        x = UpSampling2D(size=(2, 2), name="upsamp_64")(x)  # (64*64*64)

        # dec block 3
        x = Conv2DTranspose(filters=32, kernel_size=(3, 3),
                            strides=1, padding='same',
                            kernel_initializer=self.kernel_initializer,
                            activation='relu',
                            name="tconv2d_32")(x)  # (64*64*32)
        x = BatchNormalization(axis=-1)(x)
        x = UpSampling2D(size=(2, 2), name="upsamp_32")(x)  # (128*128*32)

        # dec last block
        x = Conv2DTranspose(filters=self.input_shape[2],
                            kernel_size=(3, 3),
                            kernel_initializer=self.kernel_initializer,
                            padding="same")(x)  # (128*128*3)
        output = Activation("sigmoid", name='output')(x)

        # define the encoder and AE models
        encoder = Model(inputs=inputs, outputs=latent_space)
        model = Model(inputs=inputs, outputs=output)
        return model, encoder

    def compile(self, opt="adam", loss="binary_crossentropy"):
        self.model.compile(optimizer=opt, loss=loss)

    def custom_fit(self, train, target, mlflow_experiment, representation,
                   epochs=250, batch_size=128, nb_clusters=12,
                   test=None, target_test=None, viz=True, raw_train=None,
                   preprocessing=None):
        """
        custom fit method that trains the conv
        autoencoder models with the given
         parameters
        :param train: input dataset used for training
        :param target: corresponding labels used for evaluating the model
        :param mlflow_experiment: name of the mlflow context used for training
        :param representation: feature representation name used in mlflow e.g.
        "melspectrogram" or "mfcc" etc..
        :param epochs: number of epochs used for training
        :param batch_size: batch size used for training
        :param nb_clusters: number of clusters used in K-Means
        :param test: test dataset used for testing the performance of the model
        :param target_test: corresponding labels of the test dataset
        :param viz: either to plot reduced embedded space during training
        :param raw_train: numpy array containing the raw signals loaded with
        soundFile, used for augmentation. must be not None is augmentation is True
        :param preprocessing: preprocessing class containing all needed information
        for augmentation. must be not None if augmentation is True
        :return: Keras history after using fit
        """
        if self.augmentation:
            if raw_train is None or preprocessing is None:
                print("raw_train and preprocessing must be not None, when augmentation is"
                      " True")
                return
        # update class properties
        self.nb_clusters = nb_clusters
        self.epochs = epochs
        self.batch_size = batch_size
        # define callbacks
        callbacks = [
            CustomEarlyStopping(patience=45, verbose=self.verbose, start_epoch=120),
            ReduceLROnPlateau(factor=0.1, patience=40, min_lr=1e-5, verbose=self.verbose),
            ModelCheckpoint("models/conv_ae_checkpoint.h5", verbose=self.verbose,
                            save_best_only=True, save_weights_only=True)
        ]
        if viz:
            callbacks.append(
                VisualizationCallback(self.encoder, train, target, epoch_plot=50))
        if self.augmentation:
            train_x, validation = train_test_split(raw_train, test_size=0.1, random_state=SEED)
        else:
            train_x, validation = train_test_split(train, test_size=0.1, random_state=SEED)

        start_time = datetime.now()  # save start time to compute training duration
        # set mlflow experiment name
        mlflow.set_experiment(mlflow_experiment)
        if self.verbose > 0:
            print("[INFO] mlflow experiment: {}".format(mlflow_experiment))
            if self.augmentation:
                print("[INFO] training starting with augmentation ...")
            else:
                print("[INFO] training starting ...")
        with mlflow.start_run():  # run within mlflow context
            if self.augmentation:
                train_gen = preprocessing.aug_generator(train_x,
                                                        batch_size=batch_size,
                                                        delta=1,
                                                        mode="train")
                test_gen = preprocessing.aug_generator(validation,
                                                       batch_size=batch_size,
                                                       delta=1,
                                                       mode="eval")
                history = self.model.fit(train_gen,
                                         steps_per_epoch=len(train_x) // self.batch_size,
                                         epochs=epochs,
                                         validation_data=test_gen,
                                         validation_steps=len(validation) // self.batch_size
                                         if (len(validation) // self.batch_size) >= 1 else 1,
                                         callbacks=callbacks)
            else:
                history = self.model.fit(train_x, train_x, batch_size=self.batch_size,
                                         epochs=self.epochs, callbacks=callbacks, shuffle=True,
                                         validation_data=(validation, validation))
            # log parameters to mlflow
            mlflow.log_params({
                "1.representation": representation,
                "2.latent_dim": self.latent_dim,
                "3.nb_clusters": nb_clusters,
                "4.batch_size": batch_size,
                "5.epochs": len(history.history['loss']),
                "6.input": self.input_shape
            })

            # fit and predict K-Means model,
            # then match predicted clusters with labels
            x_encoded = self.encoder.predict(train)
            self.kmeans_fit(x_encoded)
            predicted = self.kmeans.predict(x_encoded)
            new_predicted, map_cluster_to_labels = \
                match_labels_and_clusters(target, predicted)
            # update mapping clusters labels
            self.set_mapping_cluster_labels(map_cluster_to_labels, self.highest_io_class)
            if test is not None and target_test is not None:
                test_encoded = self.encoder.predict(test)
                test_kmeans = self.kmeans.predict(test_encoded)
                new_test = np.array([self.map_cluster_to_labels[str(elem)]
                                     for elem in test_kmeans])
                mlflow.log_metrics({
                    "8.f1_score_t": np.round(f1_score(target_test, new_test, average='macro'), 3),
                    "9.acc_t": np.round(acc(target_test, new_test), 3)
                })
            # log metrics to mlflow
            mlflow.log_metrics({
                "1.val_loss": np.round(min(history.history['val_loss']), 3),
                "2.f1_score": np.round(f1_score(target, new_predicted, average='macro'), 3),
                "3.recall": np.round(recall_score(target, new_predicted, average='macro'), 3),
                "4.precision": np.round(precision_score(target, new_predicted, average='macro'), 3),
                "5.acc": np.round(acc(target, new_predicted), 3),
                "6.nmi": np.round(nmi(target, predicted), 3),
                "7.ari": np.round(ari(target, predicted), 3)
            })

            # save scatter plot of latent space after applying t-sne
            X_2d = self.__apply_tsne(x_encoded)
            fig1 = self.__get_scatter_plot(data=X_2d, labels=target,
                                           legend_name='labels')
            fig1.savefig("scatter.png")

            # save scatter plot of kmeans clusters
            fig2 = self.__get_scatter_plot(data=X_2d, labels=new_predicted,
                                           legend_name='clusters')
            fig2.savefig("kmeans_clusters.png")

            # log artifacts to mlflow
            mlflow.log_artifact("scatter.png")
            mlflow.log_artifact("kmeans_clusters.png")
            # remove artifacts after logging
            os.remove("scatter.png")
            os.remove("kmeans_clusters.png")

        end_time = datetime.now()
        if self.verbose > 0:
            print('[INFO] training completed ... duration: {}'
                  .format(end_time - start_time))
        return history

    def __get_scatter_plot(self, data, labels, legend_name):
        """

        :param data:
        :param labels:
        :param legend_name:
        :return:
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

        :param data:
        :return:
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

    def kmeans_fit(self, encoded_input):
        kmeans_model = KMeans(n_clusters=self.nb_clusters, random_state=SEED)
        self.kmeans = kmeans_model.fit(encoded_input)

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

    def custom_save(self, model_path, model_name, kmeans_name, properties_name):
        """
        saving of (Keras-)model given the model file name and
        properties file name
        :param model_path: directory where to store the model
        :param model_name: keras model name e.g. convae.h5, must have .h5 as extension
        :param kmeans_name: sklearn K-Means model name saved with pickle,
        must have .pkl as extension
        :param properties_name: properties name e.g. properties.json, must have
        .json as extension
        """
        save_path = Path(PROJECT_ROOT / model_path)
        save_path.mkdir(exist_ok=True)
        self.model.save(Path(save_path / model_name))
        pickle.dump(self.kmeans, open(Path(save_path / kmeans_name), "wb"))
        prop_dict = {
            "img_height": int(self.input_shape[0]),
            "img_width": int(self.input_shape[1]),
            "nb_channels": int(self.input_shape[2]),
            "latent_dim": int(self.latent_dim),
            "nb_clusters": int(self.nb_clusters),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "kernel_initializer": self.kernel_initializer,
            "highest_io_class": int(self.highest_io_class),
            "verbose": int(self.verbose),
            "augmentation": self.augmentation
        }
        if self.map_cluster_to_labels:
            # add mapping between clusters and labels for prediction
            for key, value in self.map_cluster_to_labels.items():
                prop_dict[key] = int(value)

        with open(Path(save_path / properties_name), 'w') as fp:
            json.dump(prop_dict, fp)

    @classmethod
    def custom_load(cls, model_path, model_name, kmeans_name, properties_name):
        """
        Loading of (Keras-)model and necessary class properties given
        the model file name and properties file name
        :param model_path: directory from where to load the model and
        properties
        :param model_name: keras model file name (*.h5)
        :param kmeans_name: sklearn kmeans model name (*.pkl)
        :param properties_name: properties file name (*.json)
        :return:
        """
        load_path = Path(PROJECT_ROOT / model_path)
        with open(Path(load_path / properties_name), 'r') as fp:
            prop_dict = json.load(fp)
            convae = cls(**prop_dict)
        convae.model = load_model(Path(load_path / model_name))
        convae.encoder = Model(inputs=convae.model.input,
                               outputs=convae.model.get_layer("latent_space").output)
        convae.kmeans = pickle.load(open(Path(load_path / kmeans_name), "rb"))
        return convae

    def predict(self, samples):
        # make sure model was already trained with fit or custom_fit
        if self.kmeans is None:
            print("Model not trained yet, please train the model first.")
            return

        # transform input data into latent space - encoding
        x_encoded = self.encoder.predict(samples)
        # make prediction with kmeans
        predicted = self.kmeans.predict(x_encoded)
        if self.map_cluster_to_labels:
            predicted = np.array([self.map_cluster_to_labels[str(elem)]
                                  for elem in predicted])
        return predicted

    def custom_metrics(self, data, target):
        # make sure model was already trained with fit or custom_fit
        if self.kmeans is None:
            print("Model not trained yet, please train the model first.")
            return

        if not self.map_cluster_to_labels:
            print("please use set_mapping_cluster_labels() to define the"
                  "mapping between clusters and labels as dictionary - "
                  "cluster number as string (key) -> true label number (value)"
                  "as integer.")

        # transform input data into latent space - encoding
        x_encoded = self.encoder.predict(data)
        # make prediction with kmeans
        predicted = self.kmeans.predict(x_encoded)
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

    @classmethod
    def load(cls, model_path: str):
        convae = cls.custom_load(model_path, "conv_ae.h5",
                                 "kmeans.pkl", "properties.json")
        return convae

    def save(self, model_path: str):
        self.custom_save(model_path, "conv_ae.h5",
                         "kmeans.pkl", "properties.json")

    def fit(self, x_train: list, y_train: list = None):
        train, target = Preprocessing.dataset_from_files(io_files=x_train,
                                                         nio_files=y_train,
                                                         normal_label=NORMAL,
                                                         anomaly_label=ANOMALY,
                                                         img_size=(128, 128),
                                                         use_color=True,
                                                         reduce_noise=True)
        # define callbacks
        callbacks = [
            CustomEarlyStopping(patience=30, verbose=self.verbose, start_epoch=100),
            ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=self.verbose),
        ]
        train_x, validation = \
            train_test_split(train, target, test_size=0.05, random_state=SEED)

        start_time = datetime.now()  # save start time to compute training duration
        if self.verbose > 0:
            print("[INFO] training starting ...")
        self.model.fit(train_x, train_x, batch_size=self.batch_size,
                       epochs=self.epochs, callbacks=callbacks, shuffle=True,
                       validation_data=(validation, validation))

        end_time = datetime.now()
        if self.verbose > 0:
            print('[INFO] training completed ... duration: {}'
                  .format(end_time - start_time))

        return self

    def metrics(self, test_io: list, test_nio: list):
        # make sure model was already trained with fit or custom_fit
        if self.kmeans is None:
            print("Model not trained yet, please train the model first.")
            return

        if not self.map_cluster_to_labels:
            print("please use set_mapping_cluster_labels() to define the"
                  "mapping between clusters and labels as dictionary - "
                  "cluster number as string (key) -> true label number (value)"
                  "as integer - labels: io must be in [0, 9] and nio in [10, 19].")
            return

        test_data, test_target = Preprocessing.dataset_from_files(io_files=test_io,
                                                                  nio_files=test_nio,
                                                                  normal_label=NORMAL,
                                                                  anomaly_label=ANOMALY,
                                                                  img_size=(128, 128),
                                                                  use_color=True,
                                                                  reduce_noise=True)
        # transform input data into latent space - encoding
        x_encoded = self.encoder.predict(test_data)
        # make prediction with kmeans
        predicted = self.kmeans.predict(x_encoded)
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

        # make sure model was already trained with fit or custom_fit
        if self.kmeans is None:
            print("Model not trained yet, please train the model first.")
            return

        if not self.map_cluster_to_labels:
            print("please use set_mapping_cluster_labels() to define the"
                  "mapping between clusters and labels as dictionary - "
                  "cluster number as string (key) -> true label number (value)"
                  "as integer - labels: io must be in [0, 9] and nio in [10, 19].")
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
