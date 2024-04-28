import numpy as np
from generators import wavelet, random_pulses, power_law_noise, scale_noise


def test_wavelet():
    for _ in range(100):
        t_mu = np.random.uniform(0.1, 0.2)
        t_sigma = np.random.uniform(0.01, 0.04)
        t_min = np.random.uniform(-0.3, 0.1)
        t_max = np.random.uniform(0.2, 0.5)
        fs = np.random.randint(50, 200)
        freq = np.random.randint(1, 10)
        w, t = wavelet(t_min, t_max, fs, freq, t_mu, t_sigma)

        assert np.abs(len(w) - (t_max - t_min) * fs) < 2
        assert np.abs(t_mu - t[np.argmax(np.abs(w))]) < 5 / fs


def test_random_pulses():
    for _ in range(100):
        dur = np.random.uniform(10, 100)
        fs = np.random.randint(50, 200)
        rate = np.random.randint(1, 5)
        min_width = np.random.uniform(1 / fs * 3, 1 / fs * 5)
        max_width = np.random.uniform(1 / fs * 6, 1 / fs * 10)
        min_amp = np.random.randint(1, 3)
        max_amp = np.random.randint(4, 8)
        min_dist = 2 * max_width
        if (max_width + min_dist) * rate < 1:
            seq = random_pulses(dur, fs, rate, min_width, max_width, min_amp, max_amp)
            assert np.abs(len(seq) - (dur * fs)) < 1
        else:
            np.testing.assert_raises(
                ValueError,
                random_pulses,
                dur,
                fs,
                rate,
                min_width,
                max_width,
                min_amp,
                max_amp,
            )


def test_power_law_noise():
    dur = np.random.uniform(10, 100)
    fs = np.random.randint(50, 200)
    noise = [power_law_noise(dur, fs, alpha) for alpha in [0, 1, 2]]
    n = len(noise[0])
    assert (  # maximum power should increase
        max(np.fft.fft(noise[0])[: int(n / 2) - 1].real ** 2)
        < max(np.fft.fft(noise[1])[: int(n / 2) - 1].real ** 2)
        < max(np.fft.fft(noise[2])[: int(n / 2) - 1].real ** 2)
    )
    assert (  # total power should decrease
        sum(np.fft.fft(noise[0])[: int(n / 2) - 1].real ** 2)
        > sum(np.fft.fft(noise[1])[: int(n / 2) - 1].real ** 2)
        > sum(np.fft.fft(noise[2])[: int(n / 2) - 1].real ** 2)
    )


def test_scale_noise():
    n = np.random.randint(1000, 2000)
    noise, signal = np.random.randn(n), np.random.randn(n)
    pwr = []
    for snr in [-20, -10, 0, 10, 20]:
        pwr.append(np.mean(scale_noise(signal, noise, snr) ** 2))
    for i in range(len(pwr) - 1):
        assert np.abs(pwr[i] / pwr[i + 1] - 10) < 1e6

    pwr = []
    snr = np.random.randint(-10, 10)
    for win in np.random.randint(50, 100, 5):
        pwr.append(np.mean(scale_noise(signal, noise, snr, win) ** 2))
    assert np.max(np.abs(np.diff(pwr))) < 1e3
