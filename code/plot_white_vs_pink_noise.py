from pathlib import Path
import json
import numpy as np
from generators import power_law_noise, scale_noise
from matplotlib import pyplot as plt
import scienceplots

root = Path(__file__).parent.parent.absolute()

# configure pyplot
plt.style.use("science")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
p_plt = json.load(open(root / "code" / "plotting_parameters.json"))
plt.rc("font", size=p_plt["font"]["small"])  # controls default text sizes
plt.rc("axes", titlesize=p_plt["font"]["small"])  # fontsize of the axes title
plt.rc("axes", labelsize=p_plt["font"]["medium"])  # fontsize of the x and y labels
plt.rc("xtick", labelsize=p_plt["font"]["small"])  # fontsize of the tick labels
plt.rc("ytick", labelsize=p_plt["font"]["small"])  # fontsize of the tick labels
plt.rc("legend", fontsize=p_plt["font"]["small"])  # legend fontsize
plt.rc("figure", titlesize=p_plt["font"]["bigger"])

p = json.load(open(root / "code" / "pink_vs_white_sim_parameters.json"))
np.random.seed(p["seed"])

acc = np.load(root / "results" / "white_vs_pink_noise_sim_accuracy.npy")
reg = np.load(root / "results" / "white_vs_pink_noise_sim_lambda.npy")
reg = np.log10(reg)
n_resample = 1000

fig, ax = plt.subplots(2, 2, sharex="row", figsize=(8, 6))
for i_a, alpha in enumerate(p["alpha"]):
    noise = power_law_noise(p["dur"], p["fs"], alpha)
    times = np.linspace(0, p["dur"], len(noise))
    ax[0, i_a].plot(times, noise)
    for i_s, snr in enumerate(p["snr_db"][:-1]):
        acc_resampled = np.zeros((n_resample, len(p["dur_segment"])))
        reg_resampled = np.zeros((n_resample, len(p["dur_segment"])))
        for i_r in range(n_resample):
            idx = np.random.choice(acc.shape[0], acc.shape[0], replace=True)
            acc_resampled[i_r, :] = acc[idx, i_a, i_s, :].mean(axis=0)
            reg_resampled[i_r, :] = reg[idx, i_a, i_s, :].mean(axis=0)
        acc_resampled /= acc_resampled.max()
        ax[1, i_a].semilogx(
            p["dur_segment"],
            acc_resampled.mean(axis=0),
            label=f"{snr} dB",
            linewidth=p_plt["linewidth"],
        )
        ax[1, i_a].fill_between(
            p["dur_segment"],
            acc_resampled.mean(axis=0) - 2 * acc_resampled.std(axis=0),
            acc_resampled.mean(axis=0) + 2 * acc_resampled.std(axis=0),
            alpha=0.5,
        )
ax[1, 1].sharey(ax[1, 0])
ax[1, 0].legend(title="SNR dB", loc="lower right")
plt.subplots_adjust(wspace=0.15, hspace=0.2)
ax[0, 0].set(xlim=(0, 600), ylabel="Amplitude [a.u.]")
ax[1, 0].set(
    xlim=(min(p["dur_segment"]), max(p["dur_segment"])), ylabel="Accuracy [a.u.]"
)
ax[0, 0].set_xlabel("Time [s]", horizontalalignment="center", x=1.05)
ax[1, 0].set_xlabel("Segmend duration [s]", horizontalalignment="center", x=1.05)

for label, axes in zip(["a", "b", "c", "d"], ax.flatten()):
    axes.text(0.02, 0.91, label, transform=axes.transAxes, font="bold")

fig.savefig(root / "results" / "plots" / "white_vs_pink_noise.png", dpi=300)
