from enum import Enum


class FeatureRepresentation(Enum):
    MELSPECTROGRAM = 1
    MFCC = 2
    CQT = 3
    CHROMA_STFT = 4
    CHROMA_CENS = 5
