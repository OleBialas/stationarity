import numpy as np
from scipy import stats
from scipy.signal import correlate


def wavelet(t_min, t_max, fs, freq, t_mu, t_sigma):
    """
    Generate a wavelet (i.e. sum of a sinusiod and Gaussian) that simulates
    a neural impulse respone.

    Arguments:
        t_min (float) : start of the wavelet relative to the stimulus in seconds.
        t_max (float) : end of the wavelet relative to the stimulus in seconds.
        fs (int): sampling rate in Hz.
        freq (int): frequency of the sinusoid in Hz.
        t_mu (float): mean of the Gaussian in seconds.
        t_sigma (float): standard deviation of the Gaussian in seconds.
    Returns:
        w (np.ndarray): wavelet samples.
        t (np.ndarray): wavelet time points.
    """
    mu, sigma = fs * t_mu, fs * t_sigma
    n_min, n_max = int(t_min * fs), int(t_max * fs)
    n = n_max - n_min
    x = np.linspace(n_min, n_max, n)
    y_sine = np.sin(2 * np.pi * freq * x / fs)
    y_gauss = stats.norm.pdf(x, mu, sigma)
    w = y_sine * y_gauss
    t = x / fs
    return w, t


def random_pulses(dur, fs, rate, min_width, max_width, min_amp, max_amp):
    """
    Generate a randomized sequence of square pulses.

    Arguments:
        dur (int | float): sequeuence durarion in seconds.
        fs (int): sampling rate in Hz.
        rate (int) : average pulse rate in Hz.
        min_width (float): minimum pulse width in seconds.
        max_width (float): maximum pulse width in seconds.
        min_amp (int): minimum pulse amplitude.
        max_amp (int): maximum pulse amplitude.
    Returns:
        seq (np.ndarray): 1-dimensional array of square pulses.
    """
    min_dist = 2 * max_width
    if (max_width + min_dist) * rate > 1:
        raise ValueError("Cant generate sequence whithout violating minimum distance!")
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


def power_law_noise(dur, fs, alpha):
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

    if n_window is None:
        current_snr_dB = 10 * np.log10(np.var(signal) / np.var(noise))
        scaling_factor = 10 ** ((current_snr_dB - snr_db) / 20)
        noise *= scaling_factor
    else:
        # zero-pad with half window size
        n_pad = int(n_window / 2)
        noise = np.concatenate((np.zeros(n_pad), noise, np.zeros(n_window)))
        signal = np.concatenate((np.zeros(n_pad), signal, np.zeros(n_window)))
        for i in range(len(signal) - n_window):
            current_snr_dB = 10 * np.log10(
                np.var(signal[i : i + n_window]) / np.var(noise[i : i + n_window])
            )
            scaling_factor = 10 ** ((current_snr_dB - snr_db) / 20)
            noise[i : i + n_window] *= scaling_factor
        noise = noise[n_pad:-n_pad]
    return noise


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
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        snr = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr
        noise_scaling_factor = np.sqrt(target_noise_power / noise_power)
        scaled_noise = noise * noise_scaling_factor
    else:  # TODO: fix this
        scaled_noise = np.zeros(len(noise))
        n_pad = int(n_window / 2)
        noise = np.concatenate((np.zeros(n_pad), noise, np.zeros(n_window)))
        signal = np.concatenate((np.zeros(n_pad), signal, np.zeros(n_window)))
        for i in range(len(signal) - n_window + 1):
            signal_power = np.mean(signal[i : i + n_window] ** 2)
            noise_power = np.mean(noise[i : i + n_window] ** 2)
            snr = 10 ** (snr_db / 10)
            target_noise_power = signal_power / snr
            noise_scaling_factor = np.sqrt(target_noise_power / noise_power)
            scaled_noise[i : i + n_window] = (
                noise[i : i + n_window] * noise_scaling_factor
            )
    return scaled_noise


def simulate_response(stimulus, tf, tmin, tmax, fs):
    n_pad_start, n_pad_end = np.abs(int(tmin * fs)), np.abs(int(tmax * fs))
    response = np.zeros(len(stimulus))
    stimulus = np.concatenate([np.zeros(n_pad_start), stimulus, np.zeros(n_pad_end)])
    for i in range(len(response)):
        response[i] = np.dot(tf[::-1], stimulus[i : i + len(tf)])
    stimulus = stimulus[n_pad_start:-n_pad_end]

    stimulus = np.concatenate([np.zeros(n_pad_start), stimulus])

    response = np.convolve(stimulus, tf, mode="full")
    response = response[: len(stimulus)]
    plt.plot(np.linspace(0, len(stimulus) / fs, len(stimulus)), stimulus)
    plt.plot(np.linspace(0, len(response) / fs, len(response)), response)
    plt.show()

    response = np.convolve(stimulus, transfer_function, mode="same")
    response = response[n_pad:]
    pass
