import numpy as np


def freq_to_mel(freq):
    x = 1 + (freq / 700)
    return 2595 * np.log10(x)


def mel_to_freq(mel):
    x = 10 ** (mel / 2595)
    return 700 * (x - 1)


def construct_filter(low_freq=64, high_freq=8000, fft_points=512, num_bins=12):
    low_mel = freq_to_mel(low_freq)
    high_mel = freq_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_bins+2)
    freq_points = mel_to_freq(mel_points)
    bins = np.floor((fft_points+1) * freq_points / 44100)

    filter_bank = np.zeros((num_bins, int(np.floor(fft_points / 2 + 1))))
    for i in range(0, num_bins):
        for j in range(int(bins[i]), int(bins[i + 1])):
            filter_bank[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
        for j in range(int(bins[i + 1]), int(bins[i + 2])):
            filter_bank[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])

    return filter_bank


class ExpFilter:
    """Simple exponential smoothing filter"""
    def __init__(self, val=0.0, alpha_decay=0.5, alpha_rise=0.5):
        """Small rise / decay factors = more smoothing"""
        assert 0.0 < alpha_decay < 1.0, 'Invalid decay smoothing factor'
        assert 0.0 < alpha_rise < 1.0, 'Invalid rise smoothing factor'
        self.alpha_decay = alpha_decay
        self.alpha_rise = alpha_rise
        self.value = val

    def update(self, value):
        if isinstance(self.value, (list, np.ndarray, tuple)):
            alpha = value - self.value
            alpha[alpha > 0.0] = self.alpha_rise
            alpha[alpha <= 0.0] = self.alpha_decay
        else:
            alpha = self.alpha_rise if value > self.value else self.alpha_decay
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value
