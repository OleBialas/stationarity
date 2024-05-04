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
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from utils import segment_data, load_eeg, load_spectrogram
from mtrf import TRF
from mtrf.stats import crossval

root = Path(__file__).parent.parent.absolute()
p = json.load(open(root / "code" / "effect_of_segmentation_parameters.json"))
np.random.seed(p["seed"])


def effect_of_segmentation(sub):
    accuracy, regularization = [
        np.zeros((len(p["dur_segment"]), len(p["stim"]["n_bands"]))) for _ in range(2)
    ]
    response = load_eeg(
        sub, p["dur_train"], p["fs"], p["eeg_low_cutoff"], p["eeg_high_cutoff"]
    )
    for i_b, bands in enumerate(p["stim"]["n_bands"]):
        print(f"Computing spectrogram with {bands} bands ...")
        stimulus = load_spectrogram(bands)
        for i_d, dur in enumerate(p["dur_segment"]):
            print(f"Training TRF with {dur} second segments ...")
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
            accuracy[i_d, i_b] = r
            regularization[i_d, i_b] = trf.regularization
    np.save(root / "results" / "fit" / f"{sub}_accuracy.npy", accuracy)
    np.save(root / "results" / "fit" / f"{sub}_lambda.npy", regularization)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Fit forward TRF with variable data segment duration. Saves files {sub}_accuracy.npy and {sub}_lamba.npy in /results/fit."
    )
    parser.add_argument("sub", type=str, help="subject ID, e.g. 'sub-004'")
    args = parser.parse_args()
    effect_of_segmentation(args.sub)
