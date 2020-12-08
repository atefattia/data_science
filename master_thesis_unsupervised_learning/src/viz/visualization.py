""" script for data visualization """

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import warnings

plt.style.use('ggplot')
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

RANDOM_STATE = 123


################
#  general
################

def plot_heatmap(data_corr, annotation, figsize=(20, 10)):
    """
    convenient method for plotting a heatmap using seaborn
    :param data_corr: correlation matrix
    :param annotation: whether to display values within the heatmap
    :param figsize: figure size
    """
    fig, ax = plt.subplots(figsize=(figsize))  # Sample figsize in inches
    sns.heatmap(data_corr,
                annot=annotation,
                cmap='coolwarm',
                linewidths=.5,
                ax=ax)


def plot_pair_plot(data):
    pass


def plot_feature_maps(model_layer, image, n_square, figsize=(25, 25)):
    """
    convenient method for plotting feature maps
    :param model_layer: model layer used for displaying the
    feature map
    :param image: image used
    :param n_square: number of rows and columns (n_square*n_square)
    :param figsize: figure size
    """
    plt.figure(figsize=figsize)
    img = np.expand_dims(image, axis=0)
    # get feature map for the specified layer
    feature_maps = model_layer.predict(img)
    ix = 1
    for _ in range(n_square):
        for _ in range(n_square):
            # specify subplot and turn of axis
            ax = plt.subplot(n_square, n_square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter
            plt.imshow(feature_maps[0, :, :, ix - 1])
            ix += 1
    # show the figure
    plt.show()


def plot_reconstructed_images(model, encoder, data,
                              target, shape_enc=(50, 50),
                              figsize=(25, 8), nb_imgs=5):
    """
    convenient method for plotting input images, latent space images 
    and reconstructed images after the training
    :param model: model used to reconstruct the input imgaes
    :param encoder: encoder used to encode the input images
    :param data: contains all train data
    :param target: contains all labels
    :param shape_enc: shape used to display the encoded representation
    :param figsize: figure size
    :param nb_imgs: number of image to display
    """
    plt.figure(figsize=figsize)
    # normalise latent space
    all_encoded = encoder.predict(data)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(all_encoded)

    # pick randomly nb_imgs from train and target to display
    index = np.random.choice(data.shape[0], nb_imgs, replace=False)
    data_random = data[index]
    target_random = target[index]
    encoded_random = all_encoded[index]
    predicted_random = model.predict(data_random)

    encoded_random = scaler.transform(encoded_random)

    # reshape if channel == 1
    if len(data_random.shape) == 4 and data_random.shape[3] == 1:
        data_random = data_random.reshape(-1,
                                          data_random.shape[1],
                                          data_random.shape[2])
        predicted_random = predicted_random.reshape(-1,
                                                    predicted_random.shape[1],
                                                    predicted_random.shape[2])

    img_number = data_random.shape[0]
    for i in range(img_number):
        # display original images
        plt.subplot(3, img_number * 2, i + 1)
        plt.imshow(data_random[i])
        plt.title(target_random[i])
        plt.axis("off")

        # display encoded images
        plt.subplot(3, img_number * 2, i + 1 + (img_number * 2))
        plt.imshow(encoded_random[i].reshape(shape_enc))
        plt.axis("off")

        # display reconstructed images
        plt.subplot(3, img_number * 2, 2 * (img_number * 2) + i + 1)
        plt.imshow(predicted_random[i])
        plt.axis("off")

    plt.show()


################
#  spectrograms
################


def plot_image(image, title=None, x_label=None,
               y_label=None, figsize=None, save_path=None):
    """
    convenient method for plotting spectrograms
    :param image: input image to be plotted
    :param title: title of the image
    :param x_label: x-axis label
    :param y_label: y-axis label
    :param figsize: figure size
    :param save_path: where to save the figure if not None
    """
    if figsize:
        plt.figure(figsize=figsize)
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.reshape(image.shape[0], image.shape[1])
    plt.imshow(image)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if save_path:
        plt.axis('off')
        plt.savefig(save_path)
    plt.show()


def plot_side_by_side(img1, img2, title1=None, title2=None,
                      axis_off=True, figsize=None, save_path=None):
    """
    convenient method for plotting two spectrograms side by side
    :param img1: first input image to be plotted
    :param img2: second input image to be plotted
    :param title1: title for first image
    :param title2: title for second image
    :param axis_off: whether to show axis
    :param figsize: figure size
    :param save_path: where to save the figure if not None
    """
    if figsize:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    if len(img1.shape) == 3 and img1.shape[2] == 1:
        img1 = img1.reshape(img1.shape[0], img1.shape[1])
    if len(img2.shape) == 3 and img2.shape[2] == 1:
        img2 = img2.reshape(img2.shape[0], img2.shape[1])
    f.add_subplot(1, 2, 1)
    plt.imshow(img1)
    if title1:
        plt.title(title1)
    if axis_off:
        plt.axis('off')
    f.add_subplot(1, 2, 2)
    plt.imshow(img2)
    if title2:
        plt.title(title2)
    if axis_off:
        plt.axis('off')
    if save_path:
        plt.axis('off')
        plt.savefig(save_path)
    plt.show()


################
#  scatter
################


def plot_scatter2d(data, labels, title=None, figsize=(12, 8),
                   xlabel=None, ylabel=None):
    """
    convenient method for plotting a 2 dimensional scatterplot
    :param data: input data to be used
    :param labels: corresponding labels
    :param title: title of the scatter plot
    :param figsize: figure size
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    """
    assert data.shape[0] == labels.shape[0]
    assert data.shape[1] == 2
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels,
                         cmap=plt.cm.Paired, s=40)
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="labels")
    ax.add_artist(legend1)
    if title:
        plt.title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.show()


def plot_scatter2d_with_markers(data, labels, io_label,
                                nio_label, figsize=(12, 8),
                                title=None, xlabel=None, ylabel=None):
    """
    convenient method for plotting a 2 dimensional scatterplot  using markers for normal
    and anomalous samples
    :param data: input data to be used
    :param labels: corresponding labels
    :param io_label: list of integer containing labels of normal samples e.g. [1, 2]
    :param nio_label: list of integer containing labels of anomalous samples e.g. [3, 4]
    :param figsize: figure size
    :param title: title of the scatter plot
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    """
    assert data.shape[0] == labels.shape[0]
    assert data.shape[1] == 2
    assert labels.max() <= 12  # less than color map length
    color_map = ['blue', 'olive', 'green',
                 'red', 'deepskyblue', 'purple',
                 'orange', 'brown', 'lime',
                 'black', 'gray', 'yellow']
    colors = [color_map[label_number] for label_number in labels]
    markers = ['*' if i in io_label else '+' if i in nio_label else 'x' for i in labels]
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    for x, y, c, m in zip(data[:, 0], data[:, 1], colors, markers):
        ax.scatter(x, y, alpha=0.8, c=c, marker=m, s=70)
    if title:
        plt.title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.show()


def plot_kmeans_scatter(data, n_clusters=4, figsize=(12, 8),
                        title=None, xlabel=None,
                        ylabel=None, dim_reduction='tsne'):
    """
    convenient method for plotting a 2 dimensional scatterplot after applying K-Means
    for clustering
    :param data: input data to be used - shape (number of samples * sample features)
    :param n_clusters: number of cluster used in K-Means
    :param figsize: figure size
    :param title: title of the scatter plot
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param dim_reduction: whether to use tsne or pca to reduce the dimension for the
    scatter plot
    """
    assert dim_reduction == 'tsne' or dim_reduction == 'pca'
    if data.shape[1] == 2:
        X_2d = data
    else:  # reduce dimension to 2d with tsne or pca
        if dim_reduction == 'tsne':
            tsne2 = TSNE(n_components=2, random_state=RANDOM_STATE)
            X_2d = tsne2.fit_transform(data)
        else:  # reduce dimension with pca
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    labels = kmeans.fit(data).predict(data)
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                         cmap=plt.cm.Paired, s=40)
    # produce a legend with the unique colors from the scatter
    legend = ax.legend(*scatter.legend_elements(),
                       loc="upper right", title="clusters")
    ax.add_artist(legend)
    if title:
        plt.title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.show()
    return X_2d, labels


def plot_gmm_scatter(data, n_clusters=4, figsize=(12, 8),
                     title=None, xlabel=None,
                     ylabel=None, dim_reduction='tsne'):
    """
    convenient method for plotting a 2 dimensional scatterplot after applying Gaussian
    Mixture "GMM" for clustering
    :param data: input data to be used - shape (number of samples * sample features)
    :param n_clusters: number of cluster used in GMM
    :param figsize: figure size
    :param title: title of the scatter plot
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param dim_reduction: whether to use tsne or pca to reduce the dimension for the
    scatter plot
    """

    assert dim_reduction == 'tsne' or dim_reduction == 'pca'
    if data.shape[1] == 2:
        X_2d = data
    else:  # reduce dimension to 2d with tsne or pca
        if dim_reduction == 'tsne':
            tsne2 = TSNE(n_components=2, random_state=RANDOM_STATE)
            X_2d = tsne2.fit_transform(data)
        else:  # reduce dimension with pca
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            X_2d = pca.fit_transform(data)
    gmm = GaussianMixture(n_components=n_clusters,
                          covariance_type='full',  # tied, diag, spherical
                          max_iter=500,
                          n_init=100,
                          random_state=RANDOM_STATE,
                          )
    labels = gmm.fit_predict(data)
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                         cmap=plt.cm.Paired, s=40)
    # produce a legend with the unique colors from the scatter
    legend = ax.legend(*scatter.legend_elements(),
                       loc="upper right", title="Clusters")
    ax.add_artist(legend)
    if title:
        plt.title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.show()
    return X_2d, labels


def plot_scatter3d():
    pass


################
#  performance
################


def plot_train_history(history, figsize=(16, 10)):
    """
    Given a history of training, it plots the loss over epochs
    :param history: neural network training history
    """
    history_dict = history.history
    loss_values = history_dict['loss']
    validation_loss_values = history_dict['val_loss']
    plt.figure(figsize=figsize)
    plt.plot(loss_values, 'b', label='Training loss')
    plt.plot(validation_loss_values, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation loss')
    plt.show(block=False)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix",
                          figsize=(10, 7), cmap=plt.cm.viridis):
    """
    convenient method for plotting a confusion matrix
    :param y_true: numpy array containing the true labels
    :param y_pred: numpy array containing the predicted labels
    :param title: title of the confusion matrix
    :param figsize: figure size
    :param cmap: colormap to be used
    """
    data = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(data, columns=np.unique(y_true), index=np.unique(y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=figsize)
    with sns.plotting_context(font_scale=1.4):
        sns.heatmap(df_cm, cmap=cmap, annot=True, annot_kws={"size": 16}, fmt='g')
        plt.title(title)
        plt.show()
