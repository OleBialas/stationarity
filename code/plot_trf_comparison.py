from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt
import scienceplots
from mtrf import TRF
from mne.channels import make_standard_montage
from utils import load_eeg, load_spectrogram, segment_data

root = Path(__file__).parent.parent.absolute()
plt.style.use("science")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
p = json.load(open(root / "code" / "effect_of_segmentation_parameters.json"))

short_dur, long_dur = 5, 120  # segment durations to test
sub_id = 6  # use this subject
n_bands = 1  # model to test
thresh = 0.015  # correlation threshold
ch = "C22"  # trf channel
tmin, tmax = 0.02, 0.35  # time window to crop
rm_outliers = 0.05  # remove this percentage of outliers (with the highest values)

montage = make_standard_montage("biosemi128")

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

acc /= acc.max(axis=(1, 2), keepdims=True)  # normalize per band and subject
acc[np.isnan(acc)] = 0

diff = acc[:, short_idx, band_idx] - acc[:, long_idx, band_idx]
sub = f"sub-{str(sub_id+1).zfill(3)}"
print(f"{sub} with diff={diff[sub_id]}")
response = load_eeg(
    sub, p["dur_train"], p["fs"], p["eeg_low_cutoff"], p["eeg_high_cutoff"]
)
stimulus = load_spectrogram(n_bands)

fig, ax = plt.subplots(1, 3, sharex="row", figsize=(8, 3))
for i, (dur, idx, label) in enumerate(
    zip([short_dur, long_dur], [short_idx, long_idx], ["short", "long"])
):
    stimulus_segments, response_segments = segment_data(
        stimulus, response, p["fs"], dur, normalize=False
    )
    # compute and plot the TRF for each segment
    trfs = []
    for s, r in zip(stimulus_segments, response_segments):
        trf = TRF(preload=False)
        trf.train(s, r, p["fs"], p["tmin"], p["tmax"], reg[sub_id, idx, band_idx])
        trfs.append(trf.to_mne_evoked(montage)[0].crop(tmin, tmax).pick([ch]))
    weights = np.vstack([t.data.flatten() for t in trfs])
    ax[i].plot(trfs[0].times, weights.T, color="black", linewidth=0.1)

    # detect outliers models (with the largest absolute average weight)
    n_remove = len(response_segments) * rm_outliers
    if n_remove % 1 > 0:
        raise ValueError(
            f"Can't select {rm_outliers} from {len(response_segments)} trials"
        )
    segment_mean = np.abs(weights).mean(axis=1)
    rm_idx = np.argsort(segment_mean)[::-1][: int(n_remove)]

    # Compute the TRF on the whole data
    trf = TRF(preload=False)  # whole data model
    trf.train(
        stimulus_segments,
        response_segments,
        p["fs"],
        p["tmin"],
        p["tmax"],
        reg[sub_id, idx, band_idx],
    )
    trf = trf.to_mne_evoked(montage)[0].crop(tmin, tmax).pick([ch])
    weights = trf.data.flatten()
    ax[2].plot(trfs[0].times, weights / weights.max(), color=colors[i], label=label)

    # Compute the TRF on the whole data after removing outlier segments
    stimulus_segments = [s for i, s in enumerate(stimulus_segments) if not i in rm_idx]
    response_segments = [r for i, r in enumerate(response_segments) if not i in rm_idx]
    trf = TRF(preload=False)  # whole data model
    trf.train(
        stimulus_segments,
        response_segments,
        p["fs"],
        p["tmin"],
        p["tmax"],
        reg[sub_id, idx, band_idx],
    )
    trf = trf.to_mne_evoked(montage)[0].crop(tmin, tmax).pick([ch])
    weights = trf.data.flatten()
    ax[2].plot(trfs[0].times, weights / weights.max(), color=colors[i], linestyle="--")

ax[0].set(
    xlim=(trfs[0].times.min(), trfs[1].times.max()),
    ylabel="Model weight [a.u.]",
)
ax[1].set(xlabel="Time lag [s]")
ax[2].legend()

for label, axes in zip(["a", "b", "c"], ax.flatten()):
    axes.text(0.03, 0.95, label, transform=axes.transAxes, font="bold")

fig.savefig(root / "results" / "plots" / "trf_comparison.png", dpi=300)
