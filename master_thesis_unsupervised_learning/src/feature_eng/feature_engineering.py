import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
import librosa
from scipy.stats import skew, kurtosis
from pathlib import Path

FRAME_LENGTH = 2048
HOP_LENGTH = 512
POWER = 4
ROLL_PROCENT = 0.85
N_BANDS = 5
N_MFFCS = 5
N_CHROMA_STFT = 5
N_CHROMA_CENS = 5
N_MELS = 5
NORMALIZATION_RANGE = 10

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Feature:
    def __init__(self, csv_dir, train=None, labels=None, file_names=None,
                 sample_rate=None, recompute=False):
        """
        Constructor used to compute features from raw audio signals, then save as csv file
        or to directly load an existing csv file
        :param csv_dir: directory where csv files should be saved or loaded
        :param train: numpy array containing the raw audio signals
        :param labels: numpy array containing the corresponding labels
        :param file_names: numpy array containing the corresponding file names
        :param sample_rate: sample rate used in preprocessing to load the audio signals
        :param recompute: enforce the recomputation even when csv file exists if True,
        otherwise recompute all features
        """

        # create a directory for csv files if not exists
        self.csv_dir = Path(PROJECT_ROOT / csv_dir)
        self.csv_dir.mkdir(exist_ok=True)

        self.train = train
        self.labels = labels
        self.file_names = file_names
        self.sample_rate = sample_rate
        self.min_max_features = {}

        # define feature_names for the dateframe
        self.feature_names = ['file_name', 'label', 'audio_mean', 'audio_std', 'audio_skew',
                              'audio_kurtosis', 'audio_peak', 'zcr_mean', 'zcr_std',
                              'rms_mean', 'rms_std', 'spec_peak', 'spec_mean', 'spec_std',
                              'spec_centroid_mean', 'spec_centroid_std', 'spec_rolloff_mean',
                              'spec_rolloff_std', 'spec_bandwidth_mean', 'spec_bandwidth_std'] + \
                             ['spec_contrast_' + str(i + 1) + '_mean' for i in range(N_BANDS)] + \
                             ['spec_contrast_' + str(i + 1) + '_std' for i in range(N_BANDS)] + \
                             ['mfcc_' + str(i + 1) + '_mean' for i in range(N_MFFCS)] + \
                             ['mfcc_' + str(i + 1) + '_std' for i in range(N_MFFCS)] + \
                             ['chroma_stft_' + str(i + 1) + '_mean' for i in range(N_CHROMA_STFT)] + \
                             ['chroma_stft_' + str(i + 1) + '_std' for i in range(N_CHROMA_STFT)] + \
                             ['chroma_cens_' + str(i + 1) + '_mean' for i in range(N_CHROMA_CENS)] + \
                             ['chroma_cens_' + str(i + 1) + '_std' for i in range(N_CHROMA_CENS)] + \
                             ['melspectrogram_' + str(i + 1) + '_mean' for i in range(N_MELS)] + \
                             ['melspectrogram_' + str(i + 1) + '_std' for i in range(N_MELS)] + \
                             ['harmonic_mean', 'harmonic_std', 'percussive_mean', 'percussive_std']

        # if csv file exists AND recompute is false --> load the csv file
        # otherwise, recompute the features
        if (self.csv_dir / "df_features.csv").exists() and \
                (self.csv_dir / "df_features_normalized.csv").exists() and \
                not recompute:
            self.df = pd.read_csv(self.csv_dir / "df_features.csv")
            self.df_normalized = pd.read_csv(self.csv_dir / "df_features_normalized.csv")

        elif self.train is not None and self.labels is not None and \
                self.file_names is not None and self.sample_rate is not None:  # recompute

            self.df = self.compute_features(input_data=self.train,
                                            names=self.file_names,
                                            labels=self.labels)

            # normalize data between 0 and NORMALIZATION_RANGE
            self.df_normalized = self.normalize_df(self.df)

            # save df into csv -- before and after normalization
            self.df.to_csv(self.csv_dir / "df_features.csv", index=False)
            self.df_normalized.to_csv(self.csv_dir / "df_features_normalized.csv", index=False)
        else:
            raise AttributeError("train, labels, file_names and sample rate must be different "
                                 "from None to compute the features.")

    def compute_features(self, input_data, names, labels):
        """
        method for computing hand-crafted features
        :param input_data: numpy array of raw audio signals to be used for feature computation
        :param names: numpy array containing corresponding file names
        :param labels: numpy array containing corresponding labels
        :return: pandas dataframe of size len(input_data) * number of features
        """
        assert len(input_data) == len(names) == len(labels)

        print("-" * 30)
        print("feature computing ...")
        print("-" * 30)

        data = pd.DataFrame(columns=self.feature_names)

        for i in tqdm(range(len(input_data))):
            try:
                sample = input_data[i]
                # add the file name and the label
                feature_list = [names[i]]
                feature_list.append(labels[i])

                # ----- time domain features ----- #

                # signal: mean & standard deviation & skewness & kurtosis & peak
                feature_list.append(np.mean(abs(sample)))
                feature_list.append(np.std(sample))
                feature_list.append(skew(abs(sample)))
                # multiply by 10 to avoid FloatingPointError: underflow encountered in square
                feature_list.append(kurtosis(sample * 10))
                feature_list.append(np.max(sample))

                # zero crossing rate: mean & std
                # The zero crossing rate indicates the number of
                # times that a signal crosses the horizontal axis.
                zcr = librosa.feature.zero_crossing_rate(sample,
                                                         frame_length=FRAME_LENGTH,
                                                         hop_length=HOP_LENGTH)[0]
                feature_list.append(np.mean(zcr))
                feature_list.append(np.std(zcr))

                # root-mean-square (RMS): mean & std
                # The energy of a signal corresponds
                # to the total magnitude of the signal.
                rms = librosa.feature.rms(sample)[0]
                feature_list.append(np.mean(rms))
                feature_list.append(np.std(rms))

                # ----- frequency domain features ----- #

                # discrete Fourier Transform: peak & mean & std
                freq = np.abs(scipy.fft(sample))
                feature_list.append(freq.max())
                feature_list.append(np.mean(freq))
                feature_list.append(np.std(freq))

                # spectral centroid: mean & std
                # The spectral centroid indicates at which frequency
                # the energy of a spectrum is centered upon
                spec_centroid = librosa.feature.spectral_centroid(sample + 0.01,
                                                                  sr=self.sample_rate)[0]
                feature_list.append(np.mean(spec_centroid))
                feature_list.append(np.std(spec_centroid))

                # spectral rolloff: mean & std
                # Spectral rolloff is the frequency below which a specified percentage
                # of the total spectral energy, e.g. 85% (ROLL_PROCENT), lies.
                spec_rolloff = librosa.feature.spectral_rolloff(sample + 0.01,
                                                                sr=self.sample_rate,
                                                                roll_percent=ROLL_PROCENT)[0]
                feature_list.append(np.mean(spec_rolloff))
                feature_list.append(np.std(spec_rolloff))

                # spectral bandwidth: mean & std
                # computes the order-p (POWER) spectral bandwidth
                spec_bandwidth = librosa.feature.spectral_bandwidth(sample + 0.01,
                                                                    sr=self.sample_rate,
                                                                    p=POWER)[0]
                feature_list.append(np.mean(spec_bandwidth))
                feature_list.append(np.std(spec_bandwidth))

                # spectral contrast: mean & std
                # Spectral contrast considers the spectral peak, the spectral valley,
                # and their difference in each frequency subband
                spec_contrast = librosa.feature.spectral_contrast(sample,
                                                                  sr=self.sample_rate,
                                                                  n_bands=N_BANDS - 1,
                                                                  linear=True)
                feature_list.extend(np.mean(spec_contrast, axis=1))
                feature_list.extend(np.std(spec_contrast, axis=1))

                # ----- time-frequency domain features ----- #

                # mfcc: mean & std
                # Mel-frequency cepstral coefficients (MFCCs)
                mfcc = librosa.feature.mfcc(sample,
                                            sr=self.sample_rate,
                                            n_mfcc=5)
                feature_list.extend(np.mean(mfcc, axis=1))
                feature_list.extend(np.std(mfcc, axis=1))

                # chroma_stft: mean & std
                chroma_stft = librosa.feature.chroma_cens(sample,
                                                          sr=self.sample_rate,
                                                          n_chroma=N_CHROMA_STFT)
                feature_list.extend(np.mean(chroma_stft, axis=1))
                feature_list.extend(np.std(chroma_stft, axis=1))

                # chroma_cens: mean & std
                # chroma variant “Chroma Energy Normalized” (CENS)
                chroma_cens = librosa.feature.chroma_stft(sample,
                                                          sr=self.sample_rate,
                                                          n_chroma=N_CHROMA_CENS)
                feature_list.extend(np.mean(chroma_cens, axis=1))
                feature_list.extend(np.std(chroma_cens, axis=1))

                # melspectrogram
                # mel-scaled spectrogram
                melspectrogram = librosa.feature.melspectrogram(sample,
                                                                sr=self.sample_rate,
                                                                n_mels=N_MELS)
                feature_list.extend(np.mean(melspectrogram, axis=1))
                feature_list.extend(np.std(melspectrogram, axis=1))

                # ----- more features ----- #

                # harmonic component
                # Extract harmonic elements from an audio time-series.
                harmonic = librosa.effects.harmonic(sample)
                feature_list.append(np.mean(harmonic))
                feature_list.append(np.std(harmonic))

                # percussive component
                # Extract percussive elements from an audio time-series.
                percussive = librosa.effects.percussive(sample)
                feature_list.append(np.mean(percussive))
                feature_list.append(np.std(percussive))

                feature_list[2:] = np.round(feature_list[2:], decimals=7)
            except Exception:
                print("An error occurred while processing"
                      " the following file: '{}'".format(names[i]))
                continue

            data = data.append(pd.DataFrame(feature_list, index=self.feature_names).transpose(),
                               ignore_index=True)
        return data

    def normalize_df(self, dataframe, mode='train'):
        """
        method to normalize each feature between 0 and NORMALIZATION_RANGE
        :param dataframe: unnormalized dataframe
        :param mode: if mode is train recompute min and max for normalization,
        otherwise if mode is eval use already computed min max to normalize
        new samples
        :return: normalized dataframe
        """
        assert mode == 'train' or mode == 'eval'

        data = dataframe.copy()

        if mode == 'train':
            for col_name in self.feature_names:
                if col_name == 'file_name':
                    continue
                if col_name == 'label':
                    continue
                # save min and max per feature name for normalization of new samples
                self.min_max_features[col_name] = [data[col_name].min(), data[col_name].max()]

                data[col_name] = ((data[col_name] - data[col_name].min()) /
                                  (data[col_name].max() - data[col_name].min())) * NORMALIZATION_RANGE
                data[col_name] = data[col_name].astype(float)

        elif mode == 'eval':
            for col_name in self.feature_names:
                if col_name == 'file_name':
                    continue
                if col_name == 'label':
                    continue
                print(self.min_max_features[col_name][0])
                print(self.min_max_features[col_name][1])
                data[col_name] = NORMALIZATION_RANGE * ((data[col_name] - self.min_max_features[col_name][0]) /
                                                        (self.min_max_features[col_name][1] -
                                                         self.min_max_features[col_name][0]))

                data[col_name] = data[col_name].astype(float)

        return data

    def get_df(self, normalized=True):
        if normalized:
            return self.df_normalized
        else:
            return self.df
