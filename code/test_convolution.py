import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker


def generate_random_impulses(duration, rate, impulse_probability=0.01):
    """Generate a random series of impulses over a specified duration and sampling rate."""
    num_points = duration * rate
    impulses = np.random.rand(num_points) < impulse_probability
    return impulses.astype(int)


def causal_mexican_hat_wavelet(points, width):
    """Generate a causal Mexican hat wavelet (Ricker wavelet) of given width."""
    # Start from the center of the wavelet and extend to positive only
    full_wavelet = ricker(2 * points + 1, width)
    return full_wavelet[points:]  # Take the center to the end


def mexican_hat_wavelet(points, width):
    """Generate a Mexican hat wavelet (Ricker wavelet) of given width."""
    return ricker(points, width)


def convolve_signals(signal, kernel):
    """Convolve two signals."""
    return np.convolve(signal, kernel, mode="same")


# Parameters for the causal wavelet
# Using only the positive lags up to 300 ms (150 samples)
positive_lag_samples = int(0.3 * sampling_rate)  # 300 ms -> 150 samples

# Create a causal Mexican Hat wavelet
causal_wavelet = causal_mexican_hat_wavelet(positive_lag_samples, wavelet_width)

# Convolve impulses with the causal wavelet, ensuring only past impacts future
causal_neural_response = np.convolve(impulses, causal_wavelet, mode="full")
causal_neural_response = causal_neural_response[
    : len(impulses)
]  # Truncate to original length

# Plotting the causal neural response
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(time_vector, impulses, label="Random Impulses")
plt.title("Random Impulses over Time")
plt.ylabel("Impulse")
plt.xlabel("Time (s)")

plt.subplot(2, 1, 2)
plt.plot(time_vector, causal_neural_response, label="Causal Neural Response", color="r")
plt.title("Simulated Causal Neural Response")
plt.ylabel("Response Amplitude")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()
