"""
For each segment duration and number of spectral bands, fit a forward TRF to each subject
and compute the average prediction accuracy aross each channel

Writes:
    - for each sub, write file called `f"{sub}_accuracy.npy"` that contains a matrix
      subjects by channels `by segment duration by number of spectral bands with
      predicton accuray.
    - for each sub, write file called `f"{sub}_lambda.npy"` that contains a matrix
      subjects by segment duration by number of spectral bands. With the optimal
      value of lambda
"""

from pathlib import Path
import json
import numpy as np
from utils import segment_data, load_eeg, load_spectrogram
from mtrf import TRF
from mtrf.stats import crossval

root = Path(__file__).parent.parent.absolute()
p = json.load(open(root / "code" / "parameters.json"))
np.random.seed(p["seed"])

subjects = list((root / "data").glob("sub*"))
subjects.sort()

accuracy = np.zeros(
    (len(subjects), 128, len(p["dur_segment"]), len(p["stim"]["n_bands"]))
)
regularization = np.zeros(
    (len(subjects), len(p["dur_segment"]), len(p["stim"]["n_bands"]))
)

for i_s, sub in enumerate(subjects):
    print(f"Preparing recordings for {sub} ...")
    response = load_eeg(sub)
    for i_b, bands in enumerate(p["stim"]["n_bands"]):
        print(f"Computing spectrogram with {bands} bands ...")
        stimulus = load_spectrogram(bands)
        for i_d, dur in enumerate(p["dur_segment"]):
            print(f"Training TRF with {dur} second segments ...")
            stimulus_segments, response_segments = segment_data(
                stimulus, response, dur, normalize=False
            )
            trf = TRF()
            trf.train(
                stimulus_segments,
                response_segments,
                p["fs"],
                p["tmin"],
                p["tmax"],
                p["lambda"],
                k=p["cv_folds"],
                verbose=False,
            )
            r = crossval(
                trf,
                stimulus_segments,
                response_segments,
                p["fs"],
                p["tmin"],
                p["tmax"],
                trf.regularization,
                k=p["cv_folds"],
                average=False,
                verbose=False,
            )
            accuracy[i_s, :, i_d, i_b] = r
            regularization[i_s, i_d, i_b] = trf.regularization
np.save(root / "results" / "accuracy.npy", accuracy)
np.save(root / "results" / "lambda.npy", regularization)
