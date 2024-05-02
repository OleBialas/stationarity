from pathlib import Path
import json
import numpy as np
from mtrf import TRF
from generators import wavelet, random_pulses, power_law_noise, scale_noise
from utils import segment_data
from matplotlib import pyplot as plt

root = Path(__file__).parent.parent.absolute()
p = json.load(open(root / "code" / "mean_vs_variance_sim_parameters.json"))
np.random.seed(p["seed"])


def mean_nonstationary_noise(response, beta):
    t = np.linspace(0, p["dur"], p["dur"] * p["fs"])
    sin = beta * np.sin(2 * np.pi * p["f"] * t)
    noise = np.random.normal(sin, 1)  # noise that is variance non-stationary
    noise = scale_noise(response, noise, p["snr_db"])
    noise = np.expand_dims(noise, axis=1)
    return noise


def variance_nonstationary_noise(response, beta):
    t = np.linspace(0, p["dur"], p["dur"] * p["fs"])
    sin = np.abs(beta * np.sin(2 * np.pi * p["f"] * t))
    noise = np.random.normal(0, sin)  # noise that is variance non-stationary
    noise = scale_noise(response, noise, p["snr_db"])
    noise = np.expand_dims(noise, axis=1)
    return noise


tf, tf_times = wavelet(p["tmin"], p["tmax"], p["fs"], p["freq"], p["mu"], p["sigma"])
stim_train, stim_test = [
    random_pulses(
        p["dur"], p["fs"], p["rate"], p["wmin"], p["wmax"], p["amin"], p["amax"]
    )
    for _ in range(2)
]
stim_times = np.linspace(0, len(stim_train) / p["fs"], len(stim_train))
resp_train = np.convolve(stim_train, tf, mode="full")
resp_test = np.convolve(stim_test, tf, mode="full")
resp_train, resp_test = resp_train[: len(stim_train)], resp_test[: len(stim_test)]
stim_train = np.expand_dims(stim_train, axis=1)
stim_test = np.expand_dims(stim_test, axis=1)
resp_train = np.expand_dims(resp_train, axis=1)
resp_test = np.expand_dims(resp_test, axis=1)


for i_n, (noise_fun, name) in enumerate(
    zip([mean_nonstationary_noise, variance_nonstationary_noise], ["mean", "variance"])
):
    acc, reg = [
        np.zeros((p["n_reps"], len(p["beta"]), len(p["dur_segment"]))) for _ in range(2)
    ]
    for i_r in range(p["n_reps"]):
        print(i_r)
        for i_b, b_i in enumerate(p["beta"]):
            t = np.linspace(0, p["dur"], p["dur"] * p["fs"])
            noise = noise_fun(resp_train, b_i)
            for i_d, dur in enumerate(p["dur_segment"]):
                stim_train_segments, resp_train_segments = segment_data(
                    [stim_train], [resp_train + noise], dur, normalize=True
                )
                stim_test_segments, resp_test_segments = segment_data(
                    [stim_test], [resp_test], dur, normalize=True
                )

                trf = TRF()
                trf.train(
                    stim_train_segments,
                    resp_train_segments,
                    p["fs"],
                    p["tmin"],
                    p["tmax"],
                    p["lambda"],
                    k=p["cv_folds"],
                    seed=p["seed"],
                    verbose=False,
                )

                reg[i_r, i_b, i_d] = trf.regularization
                acc[i_r, i_b, i_d] = trf.predict(
                    stim_test_segments, resp_test_segments
                )[-1]
    np.save(root / "results" / f"{name}_nonstationary_accuracy.npy", acc)
    np.save(root / "results" / f"{name}_nonstationary_lambda.npy", reg)
