from pathlib import Path
import json
import numpy as np
import pandas as pd
from mne.io import read_raw_brainvision
from mne.channels import make_standard_montage
from mne import set_log_level
from mtrf import TRF
from slab import Sound, Filter

root = Path(__file__).parent.parent.absolute()
set_log_level("WARNING")
p = json.load(open(root / "code" / "parameters.json"))


def compute_models(stimulus, response):
    """
    Compute TRF for each stimulus/response segment and each regularization coefficient
    in the parameters.json file.

    Arguments:
        stimulus (list): stimulus feature(s) segments. Each element is a
            samples-by-features array. ELements must have the same length as the
            corresponding entry in `response`.
        response (list): neural response segments. Each element is a
            samples-by-channels array. Elements must have the same length as the
            corresponding entry in `response`.
    Returns:
        models (dict): dictionary where the keys are regularization coefficients and
        each key stores a list with a TRF model for each segment in the
        corresponding stimuli/response.
    """
    models = {f"{l: .1e}".split(".0")[1]: [] for l in p["trf"]["regularization"]}
    n_models = len(p["trf"]["regularization"]) * len(stimulus)
    count = 1
    for l in p["trf"]["regularization"]:
        for s, r in zip(stimulus, response):
            print_progress(count, n_models, prefix="Computing models:")
            count += 1
            trf = TRF()
            trf.train(s, r, p["fs"], p["trf"]["tmin"], p["trf"]["tmax"], l)
            models[f"{l: .1e}".split(".0")[1]].append(trf)
    return models


def optimize_lambda(stimulus, response, models, splits):
    """
    Find the best lambda for predicting the `stimulus` from the `response`
    using cross-validation.
    Arguments:
        stimulus (list):
        response (list):
        models (dict):
        splits(list):
    Returns
        best_lambda (str):

    """
    accuracy_per_lambda = []
    lambdas = list(models.keys())
    n_iter = len(lambdas) * len(splits)
    count = 1
    for l in lambdas:
        accuracy_per_fold = []
        for i_s, s in enumerate(splits):
            print_progress(
                count,
                n_iter,
                prefix="Optimizing lambda:",
                suffix="Complete",
                length=50,
            )
            idx_val, idx_train = s, splits[:i_s] + splits[i_s + 1 :]
            idx_train = [i for idx in idx_train for i in idx]
            model_train = np.sum([models[l][i] for i in idx_train])
            accuracy_per_fold.append(
                model_train.predict(
                    [stimulus[i] for i in idx_val],
                    [response[i] for i in idx_val],
                )[1]
            )
            count += 1
        accuracy_per_lambda.append(np.mean(accuracy_per_fold))
    return lambdas[np.argmax(accuracy_per_lambda)]


def segment_data(stimulus, response, dur, normalize=True):
    """
    Crop stimulus and response to same length and segment them into (normalized) chunks.
    Arguments:
        stimulus (list): list of samples-by-one arrays containg the stimulus feature
            for each run.
        response (list): list of samples-by-channels arrays containing EEG responses
            for each run.
        dur (int): segment duration in seconds,
        normalize (bool): if True, z-score each segment
    Returns:
        stimulus (list): list of samples-by-one arrays containg the stimulus feature
            for each segment.
        response (list): list of samples-by-channels arrays containing EEG responses
            for each segment.
    """
    n_segment = int(dur * p["fs"])
    stimulus_segments, response_segments = [], []
    for s, r in zip(stimulus, response):
        if len(s) < len(r):
            r = r[: len(s)]
        elif len(r) < len(s):
            s = s[: len(r)]
        n_remove = len(s) % n_segment
        s, r = s[n_remove:], r[n_remove:]
        if normalize:
            s = (s - s.mean(axis=0, keepdims=True)) / s.std(axis=0, keepdims=True)
            r = (r - r.mean(axis=0, keepdims=True)) / r.std(axis=0, keepdims=True)
        s = s.reshape(n_segment, -1, s.shape[-1], order="F")
        r = r.reshape(n_segment, -1, r.shape[-1], order="F")
        assert s.shape[1] == r.shape[1]
        for i_seg in range(s.shape[1]):
            stimulus_segments.append(s[:, i_seg, :])
            response_segments.append(r[:, i_seg, :])
    return stimulus_segments, response_segments


def load_spectrogram(bands):
    """
    Load spectrogram with given number of `bands` for all runs.

    Arguments:
        bands (int): number of spectral bands the signal was divided into.

    Returns:
        stimulus (list): list of arrays with dimensions samples by bands.
    """
    files = list((root / "results" / "spectrogram").glob(f"*{n}_band_spg.npy"))
    files.sort()
    if len(files) == 0:
        raise FileNotFoundError(f"Couldn't find any spectrograms with {bands} bands!")
    else:
        stimulus = [np.load(f) for f in files]
        return stimulus


def load_eeg(sub):
    """
    Load and preprocess EEG for one subject. A standard biosemi128 montage is applied,
    channels marked as bad are interpolated and the data is filtered, resampled and
    re-referenced according to the paramters.json file. Segments are cropped so
    that their combined length equals the sum of training and testing duration.
    Arguments:
        sub (str): subject identifier (e.g. "sub-004")
    Returns:
        responses (list): preprocessed EEG responses. Each element is a
            samples-by-channels array containing data from one run.
    """
    sub_folder = root / "data" / sub
    if not sub_folder.exists():
        raise ValueError(f"Can't find data for {sub}!")
    recordings = list((sub_folder / "eeg").glob("*_eeg.vhdr"))
    channels = list((sub_folder / "eeg").glob("*_channels.tsv"))
    recordings.sort()
    channels.sort()
    montage = make_standard_montage("biosemi128")
    response = []
    n_samples = int(p["dur_train"] / len(recordings) * p["fs"])
    for r, c in zip(recordings, channels):
        raw = read_raw_brainvision(r, verbose=False, preload=True)
        raw.set_montage(montage)
        bads = (pd.read_csv(c, sep="\t").status == "bad").tolist()
        raw.info["bads"] = [
            ch for ch, bad in zip(raw.info["ch_names"], bads) if bad is True
        ]
        if len(raw.info["bads"]) > 0:
            raw.interpolate_bads()
        raw.filter(p["eeg"]["low_cutoff"], p["eeg"]["high_cutoff"])
        raw.resample(p["fs"])
        raw = raw.set_eeg_reference(p["eeg"]["reference"])
        raw = raw.get_data().T
        raw = raw[:n_samples]  # crop segment
        response.append(raw)
    return response


def print_progress(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=50,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
