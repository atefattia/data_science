"""" script for extracting and processing audio files -- 100% test coverage """

import imageio
from skimage import img_as_ubyte
import numpy as np
import librosa
from glob2 import glob
from matplotlib import cm
import soundfile as sf
from skimage.transform import resize
from tqdm import tqdm
import os
import warnings
from pathlib import Path
from logmmse import logmmse
from src.data.augmentation import NoiseInjection
from src.data.feature_representation import FeatureRepresentation

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Setting seeds for reproducibility
SEED = 123

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Preprocessing:

    def __init__(self, dir_path, type="deep_learning", size=(128, 128),
                 sound_extension="flac", reduce_noise=True, save_fig=None,
                 use_color=True, noise_dir="noises/", verbose=1, mode="train"):
        """
        create an instance of the Preprocessing class
        :param dir_path: directory that contains the sound directories -
        sound directories name convention: normal samples: "train_io1", "train_io2", ..
         - anomalous samples: "train_nio1", "train_nio2", .. - unlabeled samples: "unlabeled"
        :param type: can only take three values "deep_learning" or "deep_learning_augmentation"
        for the deep learning approach with an AE or "feature_engineering" for
        a manual feature extraction
        :param size: size of the image e.g. mel spectrogram, must be square n*n
        :param sound_extension: extension used for audio e.g. .flac or .wav
        :param reduce_noise: Whether to apply denoising on the audio signal using logmmse
        :param save_fig: whether to save the created images of the audio signal
         e.g. mel spectrogram. if None don't save, otherwise save images in save_fig directory
        :param use_color: whether to use color for spectrograms
        :param noise_dir: directory that contains background noises for augmentation
        :param verbose: if greater than 0 show info messages
        :param mode: can take two values "train" for training and "test" for testing
        """
        abs_dir_path = Path(PROJECT_ROOT / dir_path)
        # check if dir_path is a directory, otherwise raise an exception
        if not os.path.isdir(abs_dir_path):
            raise NotADirectoryError("'{}' is not a directory.".format(abs_dir_path))

        # list of sound directories within dir_path
        self.dir_name = [name for name in os.listdir(abs_dir_path)
                         if os.path.isdir(Path(abs_dir_path / name))]
        self.dir_name.sort()

        # check if the number of sound directories is not 0, otherwise raise an exception
        if len(self.dir_name) == 0:
            raise FileNotFoundError("directory '{}' is empty.".format(abs_dir_path))

        if mode != "train" and mode != "test":
            raise NameError("wrong mode '{}' passed - mode only take the "
                            "following values: train or test".format(mode))

        self.mode = mode

        # check if the name of the sound directories is in the right form
        for name in self.dir_name:
            if (not name.upper().startswith((self.mode + "_io").upper())) \
                    and (not name.upper().startswith((self.mode + "_nio").upper())) \
                    and (not name.upper().startswith("unlabeled".upper())):
                raise NameError("sound directory '{}' must have this form:"
                                " '{}_io[1-9]' or '{}_nio[1-9]' or 'unlabeled'."
                                .format(name, self.mode, self.mode))

        # check if the type is correct
        if type != "deep_learning" and type != "feature_engineering"\
                and type != "deep_learning_augmentation":
            raise NameError("wrong type '{}' passed - type only take the following values: "
                            "deep_learning, deep_learning_augmentation"
                            "or feature_engineering.".format(type))

        # check if the extension is correct
        if sound_extension != "flac" and sound_extension != "wav":
            raise NameError("wrong extension '{}' passed - sound extension only "
                            "take the following values: flac or wav.".format(sound_extension))

        self.dir_path = abs_dir_path
        self.type = type
        self.img_size = size
        self.sound_extension = sound_extension
        if save_fig:
            self.save_fig = Path(PROJECT_ROOT / save_fig)
            os.makedirs(self.save_fig, exist_ok=True)
        else:
            self.save_fig = None
        self.file_names = None
        self.reduce_noise = reduce_noise
        self.use_color = use_color
        self.sample_rate = None
        self.verbose = verbose

        # set labels for each sound directory
        self.dict_name_label = {}
        label = 0
        for name in self.dir_name:
            if name.upper().startswith((self.mode + "_io").upper()):
                self.dict_name_label[name] = label
                label += 1
        for name in self.dir_name:
            if name.upper().startswith((self.mode + "_nio").upper()):
                self.dict_name_label[name] = label
                label += 1
        for name in self.dir_name:
            if name.upper() == "unlabeled".upper():
                self.dict_name_label["unlabeled"] = label

        self.io_files = {name: glob(os.path.join(self.dir_path, name, "*." + self.sound_extension))
                         for name in self.dir_name if name.upper().startswith((self.mode + "_io").upper())}
        self.nio_files = {name: glob(os.path.join(self.dir_path, name, "*." + self.sound_extension))
                          for name in self.dir_name if name.upper().startswith((self.mode + "_nio").upper())}
        self.unlabeled_files = {name: glob(os.path.join(self.dir_path, name, "*." + self.sound_extension))
                                for name in self.dir_name if name.upper() == "unlabeled".upper()}

        self.augmentation = None
        if noise_dir:
            # load and save noises for augmentation
            # many noise types can be included such as
            # background manufacturing noise
            # see aug_generator method
            abs_noise_dir = Path(PROJECT_ROOT / noise_dir)
            self.augmentation = NoiseInjection(noise_dir=abs_noise_dir)

        # print dir name and number of available files
        for dirname, all_files in self.io_files.items():
            if self.verbose > 0:
                print("[INFO] {} dir contains {} file(s)".format(dirname, len(all_files)))
            if self.save_fig:
                os.makedirs(Path(self.save_fig / dirname), exist_ok=True)
        for dirname, all_files in self.nio_files.items():
            if self.verbose > 0:
                print("[INFO] {} dir contains {} file(s)".format(dirname, len(all_files)))
            if self.save_fig:
                os.makedirs(Path(self.save_fig / dirname), exist_ok=True)
        for dirname, all_files in self.unlabeled_files.items():
            if self.verbose > 0:
                print("[INFO] {} dir contains {} file(s)".format(dirname, len(all_files)))
            if self.save_fig:
                os.makedirs(Path(self.save_fig / dirname), exist_ok=True)

    def load_sound(self, file_path):
        """
        method for loading an audio file
        :param file_path: file path for a single .flac file
        :return: data as floating point, sample rate
        """
        data, sr = sf.read(file_path)
        data = data.astype('float32')
        if self.reduce_noise:
            try:
                data = logmmse(data, sr,
                               initial_noise=1000,
                               window_size=0,
                               noise_threshold=0.15)
            except ValueError:
                pass
        if self.sample_rate is None:
            self.sample_rate = sr
        return data.astype('float32'), sr

    def get_melspectrogram(self, file_path, dirname, normalize, db_format, delta, data=None):
        """
        get mel spectrogram from file_path
        :param file_path: file path for a single .flac file
        :param dirname: name of the directory where to save the spectrograms if save_fig isn't None
        :param db_format: whether to transform spectrograms to logarithmic DB scale
        :param normalize: whether to normalize the spectrograms
        :param delta: compute the delta spectrograms according to the frequency axis "delta" times
        :return: mel spectrogram of audio data
        """
        if data is None:
            data, sr = self.load_sound(file_path)
        mel = librosa.feature.melspectrogram(data, sr=self.sample_rate)
        # compute the delta (gradient) of a spectrogram according the frequency axis
        for i in range(delta):
            mel = librosa.feature.delta(mel, axis=0, mode="nearest")
        # Convert a power spectrogram (amplitude squared) to decibel (dB) units
        if db_format:
            mel = librosa.power_to_db(mel, ref=np.max)
        # resize image
        mel = resize(mel, self.img_size, anti_aliasing=True)
        # normalize image
        if normalize:
            mel = (mel - mel.min()) / (mel.max() - mel.min())
        # a colormap instance
        if self.use_color:
            cmap = cm.jet
            mel = cmap(mel)
            mel = mel[:, :, :3]

        # save only if the save_fig path is given
        if self.save_fig:
            if delta == 0:
                save_path = "melspec_" + self.get_basename(file_path) + ".png"
            elif delta > 0:
                save_path = "melspec_d" + str(delta) + "_" + self.get_basename(file_path) + ".png"
            imageio.imwrite(os.path.join(self.save_fig, dirname, save_path), img_as_ubyte(mel))
        return mel  # if use_color==True -> shape (128*128*3) otherwise (128*128)

    def get_mfcc(self, file_path, dirname, normalize, db_format, delta=0, data=None):
        """
        get Mel Frequency Cepstral Coefficients mfcc from file_path
        :param file_path: file path for a single .flac file
        :param dirname: name of the directory where to save the spectrograms if save_fig isn't None
        :param db_format: whether to transform mfcc to logarithmic DB scale
        :param normalize: whether to normalize the mfcc
        :param delta: compute the delta spectrograms according to the frequency axis "delta" times
        :return: mfcc of audio data
        """
        if data is None:
            data, sr = self.load_sound(file_path)
        mfcc = librosa.feature.mfcc(data, sr=self.sample_rate)
        if db_format:  # Convert a power spectrogram (amplitude squared) to decibel (dB) units
            mfcc = librosa.power_to_db(mfcc, ref=np.max)
        # resize image
        mfcc = resize(mfcc, self.img_size, anti_aliasing=True)
        # normalize image
        if normalize:
            mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min())
        # a colormap instance
        if self.use_color:
            cmap = cm.jet
            mfcc = cmap(mfcc)
            mfcc = mfcc[:, :, :3]
        # save only if the save_fig path is given
        if self.save_fig:
            save_path = "mfcc_" + self.get_basename(file_path) + ".png"
            imageio.imwrite(os.path.join(self.save_fig, dirname, save_path), img_as_ubyte(mfcc))
        return mfcc  # if use_color==True -> shape (128*128*3) otherwise (128*128)

    def get_cqt(self, file_path, dirname, normalize, db_format, delta, data=None):
        """
        get the constant-Q transform CQT from file_path
        :param file_path: file path for a single .flac file
        :param dirname: name of the directory where to save the spectrograms if save_fig isn't None
        :param db_format: whether to transform CQT to logarithmic DB scale
        :param normalize: whether to normalize the CQT
        :param delta: compute the delta spectrograms according to the frequency axis "delta" times
        :return: CQT of audio data
        """
        if data is None:
            data, sr = self.load_sound(file_path)
        cqt = np.abs(librosa.cqt(data, sr=self.sample_rate))
        # compute the delta (gradient) of a spectrogram according the frequency axis
        for i in range(delta):
            cqt = librosa.feature.delta(cqt, axis=0, mode="nearest")
        if db_format:  # Convert a power spectrogram (amplitude squared) to decibel (dB) units
            cqt = librosa.power_to_db(cqt, ref=np.max)
        # resize image
        cqt = resize(cqt, self.img_size, anti_aliasing=True)
        # normalize image
        if normalize:
            cqt = (cqt - cqt.min()) / (cqt.max() - cqt.min())
        # a colormap instance
        if self.use_color:
            cmap = cm.jet
            cqt = cmap(cqt)
            cqt = cqt[:, :, :3]

        # save only if the save_fig path is given
        if self.save_fig:
            if delta == 0:
                save_path = "cqt_" + self.get_basename(file_path) + ".png"
            elif delta > 0:
                save_path = "cqt_d" + str(delta) + "_" + self.get_basename(file_path) + ".png"
            imageio.imwrite(os.path.join(self.save_fig, dirname, save_path), img_as_ubyte(cqt))
        return cqt  # if use_color==True -> shape (128*128*4) otherwise (128*128)

    def get_chroma_stft(self, file_path, dirname, normalize, db_format, delta=0, data=None):
        """
        get a chromagram stft from file_path
        :param file_path: file path for a single .flac file
        :param dirname: name of the directory where to save the spectrograms if save_fig isn't None
        :param normalize: whether to normalize the chromagram
        :param delta: compute the delta spectrograms according to the frequency axis "delta" times
        :return: chroma stft of audio data
        """
        if data is None:
            data, sr = self.load_sound(file_path)
        chroma_stft = librosa.feature.chroma_stft(y=data, sr=self.sample_rate)
        # resize image
        chroma_stft = resize(chroma_stft, self.img_size, anti_aliasing=True)
        # normalize image
        if normalize:
            chroma_stft = (chroma_stft - chroma_stft.min()) / (chroma_stft.max() - chroma_stft.min())
        # a colormap instance
        if self.use_color:
            cmap = cm.jet
            chroma_stft = cmap(chroma_stft)
            chroma_stft = chroma_stft[:, :, :3]

        # save only if the save_fig path is given
        if self.save_fig:
            save_path = "chroma_stft_" + self.get_basename(file_path) + ".png"
            imageio.imwrite(os.path.join(self.save_fig, dirname, save_path), img_as_ubyte(chroma_stft))
        return chroma_stft  # if use_color==True -> shape (128*128*4) otherwise (128*128)

    def get_chroma_cens(self, file_path, dirname, normalize, db_format, delta=0, data=None):
        """
        get the chroma variant “Chroma Energy Normalized” (CENS) from file_path
        :param file_path: file path for a single .flac file
        :param dirname: name of the directory where to save the spectrograms if save_fig isn't None
        :param normalize: whether to normalize the chroma cens
        :param delta: compute the delta spectrograms according to the frequency axis "delta" times
        :return: chroma cens of audio data
        """
        if data is None:
            data, sr = self.load_sound(file_path)
        chroma_cens = librosa.feature.chroma_cens(y=data, sr=self.sample_rate)
        # resize image
        chroma_cens = resize(chroma_cens, self.img_size, anti_aliasing=True)
        # normalize image
        if normalize:
            chroma_cens = (chroma_cens - chroma_cens.min()) / (chroma_cens.max() - chroma_cens.min())
        # a colormap instance
        if self.use_color:
            cmap = cm.jet
            chroma_cens = cmap(chroma_cens)
            chroma_cens = chroma_cens[:, :, :3]

        # save only if the save_fig path is given
        if self.save_fig:
            save_path = "chroma_cens_" + self.get_basename(file_path) + ".png"
            imageio.imwrite(os.path.join(self.save_fig, dirname, save_path), img_as_ubyte(chroma_cens))
        return chroma_cens  # if use_color==True -> shape (128*128*4) otherwise (128*128)

    def get_data(self, file_paths, feature, dirname, normalize, db_format, delta):
        """
        transform all flac sound in file_paths in a feature representation (image)
         e.g. mel spectrogram
        :param file_paths: sound paths to extract and compute
        :param feature: function to transform the audio data
        :param db_format: whether to transform spectrograms to logarithmic DB scale
        :param normalize: whether to normalize the spectrograms
        :param delta: compute the delta spectrograms according to the frequency axis "delta" times
        :return: numpy array of sound images
        """

        audio_images = [feature(filepath, dirname, normalize, db_format, delta)
                        for filepath in tqdm(file_paths)]
        files = [self.get_basename(file) for file in (file_paths)]

        # Convert to numpy array
        audio_images = np.asarray([i for i in audio_images if i is not None])
        if not self.use_color:
            audio_images = audio_images.reshape((-1, self.img_size[0], self.img_size[1], 1))
        return audio_images, files

    def shuffle_data(self, list_of_arr):
        """
        method for shuffling the content of a list of numpy array in unison
        :param list_of_arr: list of arrays with same dimension to be shuffled
        :return: a list of shuffled arrays
        """
        for i in range(len(list_of_arr) - 1):
            assert list_of_arr[0].shape[0] == list_of_arr[i + 1].shape[0]
        np.random.seed(SEED)
        index = np.arange(list_of_arr[0].shape[0])
        np.random.shuffle(index)
        for i in range(len(list_of_arr)):
            list_of_arr[i] = list_of_arr[i][index]
        return list_of_arr

    def get_deep_learning_dataset(self, feature, normalize, db_format, delta):
        """
        method for preparing the dataset for the deep learning approach
        :param feature: which feature to apply: mfcc, melspectrogram, cqt, chroma_stft,
        chroma_cens
        :param db_format: whether to transform spectrograms to logarithmic DB scale
        :param normalize: whether to normalize the spectrograms
        :param delta: compute the delta spectrograms according to the frequency axis "delta" times
        :return: two numpy arrays -- train and labels
        """
        file_names = []
        # transform io sounds
        io_images = {}
        for dirname, file_paths in self.io_files.items():
            io_images[dirname], files = self.get_data(file_paths, feature, dirname, normalize, db_format, delta)
            file_names.extend(files)

        # transform nio sounds
        nio_images = {}
        for dirname, file_paths in self.nio_files.items():
            nio_images[dirname], files = self.get_data(file_paths, feature, dirname, normalize, db_format, delta)
            file_names.extend(files)
        # transform unlabeled sounds
        unlabeled_images = {}
        for dirname, file_paths in self.unlabeled_files.items():
            unlabeled_images[dirname], files = self.get_data(file_paths, feature, dirname, normalize, db_format, delta)
            file_names.extend(files)

        # define train and labels for training
        img_shape = list(io_images.values())[0][0].shape
        train = np.empty(shape=(1, img_shape[0], img_shape[1], img_shape[2]))
        labels = []

        # concatenate io samples
        for key in io_images:
            train = np.concatenate((train, io_images[key]))
            target = np.full(len(io_images[key]), self.dict_name_label[key])
            labels.extend(target)

        # concatenate io and nio samples
        for key in nio_images:
            train = np.concatenate((train, nio_images[key]))
            target = np.full(len(nio_images[key]), self.dict_name_label[key])
            labels.extend(target)

        # concatenate io, nio, and unlabeled samples
        for key in unlabeled_images:
            train = np.concatenate((train, unlabeled_images[key]))
            target = np.full(len(unlabeled_images[key]), self.dict_name_label[key])
            labels.extend(target)

        train = train[1:]
        labels = np.asarray(labels)
        self.file_names = np.asarray(file_names)

        return train, labels

    def get_basename(self, file):
        """
        get only the basename of the path without extension -- example "document/exp.txt"
        will return "exp"
        :param file: path to the file
        :return: basename without extension
        """
        name = os.path.basename(file).split(".")[0]
        return name

    def get_feature_engineering_dataset(self):
        """
        method for preparing the dataset for the feature enginerring approach
        :return: two numpy arrays -- train and labels
        """
        file_names = []
        # read io sounds
        io_sounds = {}
        for dirname, file_paths in self.io_files.items():
            sounds = []
            for file in tqdm(file_paths):
                sound, _ = self.load_sound(file)
                sounds.append(sound)
                file_names.append(self.get_basename(file))
            io_sounds[dirname] = np.asarray(sounds)

        # read nio sounds
        nio_sounds = {}
        for dirname, file_paths in self.nio_files.items():
            sounds = []
            for file in tqdm(file_paths):
                sound, _ = self.load_sound(file)
                sounds.append(sound)
                file_names.append(self.get_basename(file))
            nio_sounds[dirname] = np.asarray(sounds)

        # read unlabeled sounds
        unlabeled_sounds = {}
        for dirname, file_paths in self.unlabeled_files.items():
            sounds = []
            for file in tqdm(file_paths):
                sound, _ = self.load_sound(file)
                sounds.append(sound)
                file_names.append(self.get_basename(file))
            unlabeled_sounds[dirname] = np.asarray(sounds)

        # define train and labels for feature engineering
        train = []
        labels = []

        # concatenate io samples
        for key in io_sounds:
            train.extend(io_sounds[key])
            target = np.full(len(io_sounds[key]), self.dict_name_label[key])
            labels.extend(target)

        # concatenate io and nio samples
        for key in nio_sounds:
            train.extend(nio_sounds[key])
            target = np.full(len(nio_sounds[key]), self.dict_name_label[key])
            labels.extend(target)

        # concatenate io, nio, and unlabeled samples
        for key in unlabeled_sounds:
            train.extend(unlabeled_sounds[key])
            target = np.full(len(unlabeled_sounds[key]), self.dict_name_label[key])
            labels.extend(target)

        train = np.asarray(train)
        labels = np.asarray(labels)
        self.file_names = np.asarray(file_names)
        return train, labels

    def get_name(self, idx):
        """
        return the file name with the index idx
        :param index: index of the file name
        :return: file name
        """
        assert idx >= 0 and idx < len(self.file_names)
        return self.file_names[idx]

    def prepare_dataset(self, feature_representation=FeatureRepresentation.MELSPECTROGRAM,
                        normalize=True, db_format=True, delta=1):
        """
        method for preparing the train and labels dataset used either for the deep learning
        or the feature engineering approach
        :param feature_representation: which sound representation to use: mfcc, melspectrogram,
        cqt, chroma_stft or chroma_cens -- enumeration
        :param db_format: whether to transform spectrograms to logarithmic DB scale
        :param normalize: whether to normalize the spectrograms
        :param delta: compute the delta spectrograms according to the frequency axis "delta" times
        :return: two numpy arrays -- train and labels
        """

        if self.verbose > 0:
            print("[INFO] data loading ...")

        feature = self.get_melspectrogram
        # deep learning approach
        if self.type == "deep_learning":
            if feature_representation == FeatureRepresentation.MFCC:
                feature = self.get_mfcc
            elif feature_representation == FeatureRepresentation.MELSPECTROGRAM:
                feature = self.get_melspectrogram
            elif feature_representation == FeatureRepresentation.CQT:
                feature = self.get_cqt
            elif feature_representation == FeatureRepresentation.CHROMA_STFT:
                feature = self.get_chroma_stft
            elif feature_representation == FeatureRepresentation.CHROMA_CENS:
                feature = self.get_chroma_cens

            train, labels = self.get_deep_learning_dataset(feature, normalize, db_format, delta)

        elif self.type == "feature_engineering":
            train, labels = self.get_feature_engineering_dataset()
        elif self.type == "deep_learning_augmentation":
            train, labels = self.get_feature_engineering_dataset()

        # shuffle data - train & labels & file_names
        shuffled_arr = self.shuffle_data([train, labels, self.file_names])
        train = shuffled_arr[0]
        labels = shuffled_arr[1]
        self.file_names = shuffled_arr[2]

        if self.verbose > 0:
            print("[INFO] train shape: ", train.shape)
            print("[INFO] labels shape: ", labels.shape)
        return train, labels

    def aug_generator(self, input_train, batch_size, delta, mode="train",
                      feature_representation=FeatureRepresentation.MELSPECTROGRAM):
        """
        Keras generator enables to augment the data with noise injection
        during the training
        :param input_train: numpy array created with the Preprocessing class
        and type == "deep_learning_augmentation"
        :param batch_size: batch size used
        :param delta: whether to compute the delta spectrograms according to
         the frequency axis "delta" times
        :param mode: take two values train for trainin and eval for evaluation
        :param feature_representation: which representation to use -- enumeration
        :return: batch of augmented train set and batch of real train set
        """
        assert self.augmentation is not None
        assert mode == "train" or mode == "eval"

        if feature_representation == FeatureRepresentation.MFCC:
            feature = self.get_mfcc
        elif feature_representation == FeatureRepresentation.MELSPECTROGRAM:
            feature = self.get_melspectrogram
        elif feature_representation == FeatureRepresentation.CQT:
            feature = self.get_cqt
        elif feature_representation == FeatureRepresentation.CHROMA_STFT:
            feature = self.get_chroma_stft
        elif feature_representation == FeatureRepresentation.CHROMA_CENS:
            feature = self.get_chroma_cens

        idx_batch_begin = 0
        # loop indefinitely
        while True:
            # define augmentation here
            noise_injection = self.augmentation.noise_injection()

            if idx_batch_begin >= len(input_train):
                idx_batch_begin = 0  # reload from the beginning

            idx_batch_end = min(idx_batch_begin + batch_size, len(input_train))
            train_target = [feature(data=input_train[i],
                                    normalize=True,
                                    db_format=True,
                                    delta=delta,
                                    file_path="",
                                    dirname="") for i in range(idx_batch_begin, idx_batch_end)]
            train_aug = [feature(data=noise_injection.augment(input_train[i]),
                                 normalize=True,
                                 db_format=True,
                                 delta=delta,
                                 file_path="",
                                 dirname="") for i in range(idx_batch_begin, idx_batch_end)]

            # next batch
            idx_batch_begin += batch_size

            # yield the batch to the calling function
            yield np.array(train_aug), np.array(train_target)

    @staticmethod
    def dataset_from_files(io_files: list, nio_files: list = None,
                           normal_label=0, anomaly_label=1, img_size=(128, 128),
                           use_color=True, reduce_noise=True, verbose=1):
        """
        convenient method for preparing data for the base_model module
        :param io_files: iterable of strings for paths to io files
        :param nio_files: iterable of strings for paths to nio files
        :param normal_label: int used to define normal samples
        :param anomaly_label: int used to define anomalous samples
        :param img_size: the input image size
        :param use_color: whether to use colors
        :param reduce_noise: whether to reduce noise with logmmse
        :param verbose: if greater than 0 show info messages
        :return: numpy array of train and label dataset
        """
        if verbose > 0:
            print("[INFO] data loading ...")

        def load_sound(file_path, reduce_noise):
            data, sr = sf.read(file_path)
            data = data.astype('float32')
            if reduce_noise:
                try:
                    data = logmmse(data, sr,
                                   initial_noise=1000,
                                   window_size=0,
                                   noise_threshold=0.15)
                except ValueError:
                    pass
            return data.astype('float32'), sr

        def get_melspectrogram(file_path, delta, img_size, use_color, reduce_noise):
            data, sr = load_sound(file_path, reduce_noise=reduce_noise)
            mel = librosa.feature.melspectrogram(data, sr=sr)
            for i in range(delta):
                mel = librosa.feature.delta(mel, axis=0, mode="nearest")
            mel = librosa.power_to_db(mel, ref=np.max)
            mel = resize(mel, img_size, anti_aliasing=True)
            mel = (mel - mel.min()) / (mel.max() - mel.min())
            # a colormap instance
            if use_color:
                cmap = cm.jet
                mel = cmap(mel)
                mel = mel[:, :, :3]
            return mel  # if use_color==True -> shape (128*128*3) otherwise (128*128)

        io_audio_images = [get_melspectrogram(filepath,
                                              delta=1,
                                              img_size=img_size,
                                              use_color=use_color,
                                              reduce_noise=reduce_noise)
                           for filepath in tqdm(io_files)]

        # Convert to numpy array
        io_audio_images = np.asarray([i for i in io_audio_images if i is not None])
        if not use_color:
            io_audio_images = io_audio_images.reshape((-1, img_size[0], img_size[1], 1))
        io_target = np.full(len(io_audio_images), normal_label)

        nio_audio_images = []
        nio_target = []
        if nio_files:
            nio_audio_images = [get_melspectrogram(filepath,
                                                   delta=1,
                                                   img_size=img_size,
                                                   use_color=use_color,
                                                   reduce_noise=reduce_noise)
                                for filepath in tqdm(nio_files)]
            # Convert to numpy array
            nio_audio_images = np.asarray([i for i in nio_audio_images if i is not None])
            if not use_color:
                nio_audio_images = nio_audio_images.reshape((-1, img_size[0], img_size[1], 1))
            nio_target = np.full(len(nio_audio_images), anomaly_label)

        if nio_files:
            train = np.concatenate((io_audio_images, nio_audio_images))
            labels = np.concatenate((io_target, nio_target))
        else:
            train = io_audio_images
            labels = io_target

        train = np.asarray(train)
        labels = np.asarray(labels)

        # shuffle data
        np.random.seed(SEED)
        index = np.arange(train.shape[0])
        np.random.shuffle(index)
        train = train[index]
        labels = labels[index]

        if verbose > 0:
            print("[INFO] data shape: ", train.shape)
            print("[INFO] labels shape: ", labels.shape)

        return train, labels
