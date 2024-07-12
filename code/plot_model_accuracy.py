from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel
import scienceplots

root = Path(__file__).parent.parent.absolute()
plt.style.use("science")
p = json.load(open(root / "code" / "effect_of_segmentation_parameters.json"))

thresh = 0.01  # ignore when accuracy is below this threshold
n_resample = 1000
short_dur, long_dur = 5, 120  # segment durations for comparing accuracy
n_bands = 16  # model for comparing accuracy
short_idx = np.where(np.asarray(p["dur_segment"]) == short_dur)[0][0]
long_idx = np.where(np.asarray(p["dur_segment"]) == long_dur)[0][0]
band_idx = np.where(np.asarray(p["stim"]["n_bands"]) == n_bands)[0][0]

acc_files = list((root / "results" / "fit").glob("*_accuracy.npy"))
reg_files = list((root / "results" / "fit").glob("*_lambda.npy"))
acc_files.sort()
reg_files.sort()

acc = np.zeros((len(acc_files), len(p["dur_segment"]), len(p["stim"]["n_bands"])))
reg = np.zeros((len(reg_files), len(p["dur_segment"]), len(p["stim"]["n_bands"])))
for i_sub, (acc_file, reg_file) in enumerate(zip(acc_files, reg_files)):
    acc_sub = np.load(acc_file)
    reg_sub = np.load(reg_file)
    if acc_sub.max() > thresh:
        acc[i_sub, :, :] = acc_sub
        reg[i_sub, :, :] = reg_sub


sub_acc = acc[:, :, band_idx].max(axis=1)
mask = acc.sum(axis=(1, 2)) == 0  # remove 0 entries
print(f"removing {sum(mask)} elements due to correlation threshold")
acc = acc[np.invert(mask)]

# paired t-test on prediction accuracy
print(acc[:, short_idx, band_idx].mean())
print(acc[:, short_idx, band_idx].std())
print(acc[:, long_idx, band_idx].mean())
print(acc[:, long_idx, band_idx].std())
print(ttest_rel(acc[:, short_idx, band_idx], acc[:, long_idx, band_idx]))

sub_acc = sub_acc[np.invert(mask)]
reg = reg[reg.sum(axis=(1, 2)) != 0]
acc /= acc.max(axis=(1, 2), keepdims=True)  # normalize per band and subject
diff = acc[:, short_idx, band_idx] - acc[:, long_idx, band_idx]
reg = np.log10(reg)  # log-transform regularization coefficients

# bootstrap resampling
acc_resampled = np.zeros((n_resample, len(p["dur_segment"]), len(p["stim"]["n_bands"])))
reg_resampled = np.zeros((n_resample, len(p["dur_segment"]), len(p["stim"]["n_bands"])))
for i in range(n_resample):
    idx = np.random.choice(acc.shape[0], acc.shape[0], replace=True)
    acc_resampled[i, :, :] = acc[idx].mean(axis=0)
    reg_resampled[i, :, :] = reg[idx].mean(axis=0)

fig, ax = plt.subplots(1, 3, figsize=(8, 3))

ax[2].scatter(sub_acc, diff, color="black")
ax[2].set(xlabel="Model accuracy [r]", ylabel="Change in accuracy [a.u.]")

mean, std = acc_resampled.mean(axis=0), acc_resampled.std(axis=0)
for i_n, n in enumerate(p["stim"]["n_bands"]):
    if i_n < 3:
        ax[0].semilogx(p["dur_segment"], mean[:, i_n], label=n)
        ax[0].fill_between(
            p["dur_segment"],
            mean[:, i_n] + std[:, i_n],
            mean[:, i_n] - std[:, i_n],
            alpha=0.4,
        )

mean, std = reg_resampled.mean(axis=0), reg_resampled.std(axis=0)
for i_n, n in enumerate(p["stim"]["n_bands"]):
    if i_n < 3:
        ax[1].semilogx(p["dur_segment"], mean[:, i_n], label=n)
        ax[1].fill_between(
            p["dur_segment"],
            mean[:, i_n] + std[:, i_n],
            mean[:, i_n] - std[:, i_n],
            alpha=0.4,
        )

ax[0].legend(title="spectral bands")
ax[0].set(
    xlim=(p["dur_segment"][0], p["dur_segment"][-1]),
    ylabel="Accuracy [a.u.]",
    xlabel="Segment duration [s]",
    ylim=(0.75, 0.96),
)
ax[1].set(ylabel="Regularization [a.u.]", xlabel="Segment duration [s]")

for label, axes in zip(["a", "b", "c"], ax.flatten()):
    axes.text(0.03, 0.95, label, transform=axes.transAxes, font="bold")

plt.tight_layout()
fig.savefig(root / "results" / "plots" / "model_accuracy.png", dpi=300)
