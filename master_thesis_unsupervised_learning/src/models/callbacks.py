""" script for defining custom keras callbacks """

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

SEED = 123


class CustomEarlyStopping(EarlyStopping):

    def __init__(self, monitor='val_loss', min_delta=0,
                 patience=0, verbose=0,
                 mode='auto', start_epoch=100):
        """
        make early stopping happen only after start_epoch
        :param start_epoch: number of epochs after that early stopping
        can occur
        """
        super().__init__(monitor=monitor, min_delta=min_delta,
                         patience=patience, mode=mode,
                         verbose=verbose)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


class VisualizationCallback(Callback):

    def __init__(self, encoder, samples, target, epoch_plot):
        super().__init__()
        self.samples = samples
        self.encoder = encoder
        self.target = target
        self.epoch_plot = epoch_plot

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_plot == 0:
            x_encoded = self.encoder.predict(self.samples)

            restore_these_settings = np.geterr()

            temp_settings = restore_these_settings.copy()
            temp_settings["over"] = "ignore"
            temp_settings["under"] = "ignore"

            np.seterr(**temp_settings)
            tsne2 = TSNE(n_components=2, random_state=SEED)
            X_2d = tsne2.fit_transform(x_encoded)
            np.seterr(**restore_these_settings)
            plt.figure(figsize=(10, 7))
            plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.target,
                        label=self.target, cmap=plt.cm.Paired)
            plt.show()
