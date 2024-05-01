from pathlib import Path
import json
import numpy as np
from generators import power_law_noise, scale_noise
from matplotlib import pyplot as plt
import scienceplots

root = Path(__file__).parent.parent.absolute()
plt.style.use("science")
p = json.load(open(root / "code" / "pink_vs_white_sim_parameters.json"))
np.random.seed(p["seed"])

acc = np.load(root / "results" / "white_vs_pink_noise_sim_accuracy.npy")
reg = np.load(root / "results" / "white_vs_pink_noise_sim_lambda.npy")
reg = np.log10(reg)
n_resample = 1000

fig, ax = plt.subplots(2, 3, sharex="row", sharey="row")
for i_a, alpha in enumerate(p["alpha"]):
    noise = power_law_noise(p["dur"], p["fs"], alpha)
    times = np.linspace(0, p["dur"], len(noise))
    ax[0, i_a].plot(times, noise)
    acc_resampled = np.zeros((n_resample, len(p["dur_segment"])))
    reg_resampled = np.zeros((n_resample, len(p["dur_segment"])))
    for i_r in range(n_resample):
        idx = np.random.choice(acc.shape[0], acc.shape[0], replace=True)
        acc_resampled[i_r, :] = acc[idx, i_a, 0, :].mean(axis=0)
        reg_resampled[i_r, :] = reg[idx, i_a, 0, :].mean(axis=0)
    acc_resampled /= acc_resampled.max()
    ax[1, i_a].semilogx(p["dur_segment"], acc_resampled.mean(axis=0))
    ax[1, i_a].fill_between(
        p["dur_segment"],
        acc_resampled.mean(axis=0) - acc_resampled.std(axis=0),
        acc_resampled.mean(axis=0) + acc_resampled.std(axis=0),
        alpha=0.5,
    )
    ax2 = ax[1, i_a].twinx()
    ax2.semilogx(p["dur_segment"], reg_resampled.mean(axis=0))
