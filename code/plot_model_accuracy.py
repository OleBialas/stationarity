from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt
import scienceplots

root = Path(__file__).parent.parent.absolute
plt.style.use("science")
p = json.load(open(root / "code" / "effect_of_segmentation_parameters.json"))

dur_segments = [1, 10, 30, 60, 120]
n_bands = [1, 16, 32]
thresh = 0.01  # ignore when accuracy is below this threshold
n_resample = 1000

acc_files = list((root / "results" / "fit").glob("*_accuracy.npy"))
reg_files = list((root / "results" / "fit").glob("*_lambda.npy"))
acc_files.sort()
reg_files.sort()

acc = np.zeros((len(acc_files), len(dur_segments), len(n_bands)))
reg = np.zeros((len(reg_files), len(dur_segments), len(n_bands)))
for i_sub, (acc_file, reg_file) in enumerate(zip(acc_files, reg_files)):
    acc_sub = np.load(acc_file)
    reg_sub = np.load(reg_file)
    if acc_sub.max() > thresh:
        acc[i_sub, :, :] = acc_sub
        reg[i_sub, :, :] = reg_sub

# remove 0 entries
acc = acc[acc.sum(axis=(1, 2)) != 0]
reg = reg[reg.sum(axis=(1, 2)) != 0]

acc /= acc.max(axis=1, keepdims=True)  # normalize per band and subject
reg = np.log10(reg)  # log-transform regularization coefficients

# bootstrap resampling
acc_resampled = np.zeros((n_resample, len(dur_segments), len(n_bands)))
reg_resampled = np.zeros((n_resample, len(dur_segments), len(n_bands)))
for i in range(n_resample):
    idx = np.random.choice(acc.shape[0], acc.shape[0], replace=True)
    acc_resampled[i, :, :] = acc[idx].mean(axis=0)
    reg_resampled[i, :, :] = reg[idx].mean(axis=0)

fig, ax = plt.subplots(1, 2, sharex=True, figsize=(8, 6))

mean, std = acc_resampled.mean(axis=0), acc_resampled.std(axis=0)
for i_n, n in enumerate(n_bands):
    ax[0].plot(dur_segments, mean[:, i_n], label=n)
    ax[0].fill_between(
        dur_segments, mean[:, i_n] + std[:, i_n], mean[:, i_n] - std[:, i_n], alpha=0.5
    )

mean, std = reg_resampled.mean(axis=0), reg_resampled.std(axis=0)
for i_n, n in enumerate(n_bands):
    ax[1].plot(dur_segments, mean[:, i_n], label=n)
    ax[1].fill_between(
        dur_segments, mean[:, i_n] + std[:, i_n], mean[:, i_n] - std[:, i_n], alpha=0.5
    )

ax[0].legend(title="spectral bands")
ax[0].set(
    xlim=(dur_segments[0], dur_segments[-1]), ylim=(0.9, 1), ylabel="Accuracy [a.u.]"
)
ax[1].set(ylabel="Regularization [a.u.]")
fig.supxlabel("Segment duration [s]")

for label, axes in zip(["a", "b"], ax.flatten()):
    axes.text(0.025, 0.92, label, transform=axes.transAxes, font="bold")

fig.savefig(root / "results" / "plots" / "model_accuracy.png", dpi=300)
