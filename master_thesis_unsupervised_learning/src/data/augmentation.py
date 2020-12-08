""" script for injecting noise, such as manufacturing
background noise, to make models more robust """

import nlpaug.augmenter.audio as naa
import os
from glob2 import glob
import soundfile as sf
import numpy as np
import random


class NoiseInjection:

    def __init__(self, noise_dir, clean_noise_ratio=2):
        """
        define a list of clean and noisy samples used afterwards for augmentation
        :param noise_dir: directory that contains the noise samples
        :param clean_noise_ratio: for each noisy sample add "clean_noise_ratio"
        times clean samples, for example for 10 noisy samples and clean_noise_ratio
        == 2, aug_noise will contain 10 noisy and 20 clean samples.
        this will randomly generate for each batch clean batches and augmented
        batches.
        """
        self.noise_dir = noise_dir
        self.clean_noise_ratio = clean_noise_ratio
        self.aug_noises = []
        self.load_noises()

    def load_noises(self):
        """
        load noisy samples from the specified directory
        """
        noises = glob(os.path.join(self.noise_dir, '*.flac'))
        self.aug_noises = []
        for file in noises:
            noise, _ = sf.read(file)
            self.aug_noises.append(noise)
            clean_sample = np.zeros_like(noise)
            for i in range(self.clean_noise_ratio):
                self.aug_noises.append(clean_sample)

    def noise_injection(self):
        """
        define a noise augmentation instance using nlpaug
        :return: noise augmentation instance
        """
        # pick randomly k samples from aug_noises
        sampling = random.choices(self.aug_noises, k=3)
        # define instance for noise augmentation
        noise_injection = naa.NoiseAug(zone=(0.0, 1.0),
                                       noises=sampling)
        return noise_injection
