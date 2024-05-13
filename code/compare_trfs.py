from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt
import scienceplots
from utils import load_eeg, load_spectrogram

root = Path(__file__).parent.parent.absolute()
plt.style.use("science")
p = json.load(open(root / "code" / "effect_of_segmentation_parameters.json"))

# find subjects for which the relative difference in accuracy
# between short and long segments is greatest
short_dur, long_dur = 5, 120
n_bands = 1
thresh = 0.015

short_idx = np.where(np.asarray(p["dur_segment"]) == short_dur)[0][0]
long_idx = np.where(np.asarray(p["dur_segment"]) == long_dur)[0][0]
band_idx = np.where(np.asarray(p["stim"]["n_bands"]) == n_bands)[0][0]

acc_files = list((root / "results" / "fit").glob("*_accuracy.npy"))
acc_files.sort()

acc = np.zeros((len(acc_files), len(p["dur_segment"]), len(p["stim"]["n_bands"])))
for i_sub, acc_file in enumerate(acc_files):
    acc_sub = np.load(acc_file)
    if acc_sub.max() > thresh:
        acc[i_sub, :, :] = acc_sub

acc /= acc.max(axis=(1, 2), keepdims=True)  # normalize per band and subject
acc[np.isnan(acc)] = 0

diff = acc[:, short_idx, band_idx] - acc[:, long_idx, band_idx]
sub_id = np.argmax(diff) + 1

sub = f"sub-{str(sub_id).zfill(3)}"
response = load_eeg(
    sub, p["dur_train"], p["fs"], p["eeg_low_cutoff"], p["eeg_high_cutoff"]
)
stimulus = load_spectrogram(n_bands)

for dur in [short, long]
    stimulus_segments, response_segments = segment_data(
        stimulus, response, p["fs"], dur, normalize=False
    )
    trf = TRF(preload=False)
    r = trf.train(
        stimulus_segments,
        response_segments,
        p["fs"],
        p["tmin"],
        p["tmax"],
        p["lambda"],
        k=p["cv_folds"],
        verbose=False,
    ).max()
