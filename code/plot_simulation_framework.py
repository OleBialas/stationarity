from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt
import scienceplots
from mtrf import TRF
from mtrf.stats import neg_mse
from generators import wavelet, random_pulses, scale_noise

root = Path(__file__).parent.parent.absolute()
plt.style.use("science")
p = json.load(open(root / "code" / "parameters.json"))
np.random.seed(p["seed"])

fs = 250
dur = 1200
tmin, tmax = 0, 0.4

tf, tf_times = wavelet(tmin, tmax, fs, 4, 0.1, 0.03)
stim = random_pulses(dur, fs, 3, 0.005, 0.050, 1, 5)
resp = np.convolve(stim, tf, mode="full")
resp = resp[: len(stim)]
stim_times = np.linspace(0, len(stim) / fs, len(stim))

noise = np.random.randn(len(stim))
noise = scale_noise(resp, noise, -20)

trf = TRF()
r = trf.train(
    np.array_split(stim, 5),
    np.array_split(resp + noise, 5),
    fs,
    -0.1,
    0.5,
    p["lambda"],
).max()

pred = trf.predict(stim)[0].flatten()

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ax[0, 0].plot(tf_times, tf / tf.max(), label="original")
ax[0, 0].plot(
    trf.times,
    trf.weights.flatten() / trf.weights.flatten().max(),
    label="reconstruction",
)
ax[0, 1].plot(stim_times, stim, label="stimulus")
ax[0, 1].plot(stim_times, (resp / resp.max()) * stim.max(), label="response")
ax[1, 0].plot(stim_times, resp + noise)
ax[1, 1].plot(stim_times, pred, label="prediction")
ax[1, 1].plot(stim_times, resp, label="response")
ax[0, 0].set(yticks=[])
ax[0, 1].set(xlim=(0, 5), yticks=[])
ax[1, 0].set(xlim=(0, 5), yticks=[])
ax[1, 1].set(xlim=(0, 5), yticks=[])
ax[0, 0].legend(loc="upper right")
ax[0, 1].legend(loc="upper right")
ax[1, 1].legend(loc="upper right")

for label, axes in zip(["a", "b", "c", "d"], ax.flatten()):
    axes.text(0.01, 0.92, label, transform=axes.transAxes, font="bold")

fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
plt.xlabel("Amplitude [a.u.]")
plt.ylabel("Time [s]")

plt.tight_layout()
fig.savefig(root / "results" / "plot" / "simulation_framework.png", dpi=300)
