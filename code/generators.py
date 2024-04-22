import numpy as np
from scipy import stats
from scipy.signal import correlate


def wavelet(t_min, t_max, fs, freq, t_mu, t_sigma):
    mu, sigma = int(fs * t_mu), int(fs * t_sigma)
    n_min, n_max = int(t_min * fs), int(t_max * fs)
    n = n_max - n_min
    x = np.linspace(n_min, n_max, n)
    y_sine = np.sin(2 * np.pi * freq * x / fs)
    y_gauss = stats.norm.pdf(x, mu, sigma)
    return y_sine * y_gauss, x / fs


def random_pulses(dur, fs, rate, min_width, max_width, min_amp, max_amp):
    """
    Generate a randomized sequence of square pulses.

    Arguments:
        dur (int | float): sequeuence durarion in seconds.
        fs (int): sampling rate in Hz.
        rate (int) : average pulse rate in Hz.
        min_width (int): minimum pulse width in samples.
        max_width (int): maximum pulse width in samples.
        min_amp (int): minimum pulse amplitude.
        max_amp (int): maximum pulse amplitude.
    Returns:
        seq (np.ndarray): 1-dimensional array of square pulses.
    """
    min_dist = 2 * max_width
    n_samples = int(dur * fs)
    n_pulses = int(dur * rate)
    n_dist = int(min_dist * fs)
    seq = np.zeros(n_samples)
    n_min_width, n_max_width = int(fs * min_width), int(fs * max_width)
    idx = np.random.choice(n_samples, n_pulses, replace=False)
    idx.sort()
    idx = idx[np.diff(idx, prepend=True) > n_dist]
    n_remaining = n_pulses - len(idx)
    while n_remaining > 0:
        i = np.random.choice(n_samples, 1)
        if np.abs(idx - i).min() >= n_dist:
            idx = np.concatenate([idx, i])
            n_remaining -= 1
    idx.sort()
    for i in idx:
        w = int(np.random.randint(n_min_width, n_max_width) / 2)
        seq[i - w : i + w] = np.random.randint(min_amp, max_amp)
    return seq


def powerlawnoise(dur, fs, alpha, return_spectrum=False):
    """
    Generate noise and impose a 1/f^alpha distribution on its power spectrum.

    Arguments:
        dur (int | float): sequeuence durarion in seconds.
        fs (int): sampling rate in Hz.
        alpha (int | float): spectral scaling coefficient.
    Returns:
        noise (np.ndarray): generated noise waveform.
    """
    n = int(dur * fs)
    fft = np.fft.fft(np.random.normal(0, 1, n))
    freq = np.fft.fftfreq(n, 1 / fs)
    psd = np.concatenate([np.ones(1), 1 / np.abs(freq[1:]) ** alpha])
    fft *= np.sqrt(psd)
    pwr = np.sum(np.abs(fft) ** 2) / (fs * n)
    scaling_factor = np.sqrt(len(fft) / pwr)
    fft *= scaling_factor
    noise = np.fft.ifft(fft).real
    return noise


def autocorrelation(signal):
    """
    Return the signals normalized autocorrelation
    """
    r = correlate(signal, signal, mode="full")
    r = r / np.max(r)
    return r[len(r) // 2 :]


def scale_noise(signal, noise, snr_db, n_window=None):
    """
    Scale the noise so that the sum of signal and noise has given SNR.

    Arguments:
        signal (np.ndarray): 1-dimensional array containing the signal.
        noise (np.ndarray): 1-dimensional array containing the noise.
        n_window (None | int): if not None, scale the noise in a sliding
            window of the given size.
    Returns:
        noise (np.ndarray): scaled noise.
    """
    if n_window is None:
        current_snr_dB = 10 * np.log10(np.var(signal) / np.var(noise))
        scaling_factor = 10 ** ((current_snr_dB - snr_db) / 20)
        noise *= scaling_factor
    else:
        noise = np.concatenate((np.zeros(n_window), noise, np.zeros(n_window)))
        signal = np.concatenate((np.zeros(n_window), signal, np.zeros(n_window)))
        for i in range(len(signal) - n_window):
            current_snr_dB = 10 * np.log10(
                np.var(signal[i : i + n_window]) / np.var(noise[i : i + n_window])
            )
            scaling_factor = 10 ** ((current_snr_dB - snr_db) / 20)
            noise[i : i + n_window] *= scaling_factor
        noise = noise[n_window:-n_window]
    return noise
