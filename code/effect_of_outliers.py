from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt
from mtrf import TRF
from mne.channels import make_standard_montage
from utils import load_eeg, load_spectrogram, segment_data

root = Path(__file__).parent.parent.absolute()
p = json.load(open(root / "code" / "effect_of_outliers_parameters.json"))

stimulus = load_spectrogram(p["stim"]["n_bands"])

for sub in (root / "data").glob("sub-*"):
    response = load_eeg(
        sub, p["dur_train"], p["fs"], p["eeg_low_cutoff"], p["eeg_high_cutoff"]
    )

    stimulus_segments, response_segments = segment_data(
        stimulus, response, p["fs"], p["dur_train"], normalize=False
    )
