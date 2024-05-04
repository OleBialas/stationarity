from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt
from mtrf import TRF
from mtrf.stats import pearsonr, neg_mse
from generators import (
    wavelet,
    random_pulses,
    power_law_noise,
    scale_noise,
    jitter_pulses,
)

root = Path(__file__).parent.parent.absolute()
p = json.load(open(root / "code" / "confound_sim_parameters.json"))
np.random.seed(p["seed"])

tf, tf_times = wavelet(  # generative impluse response
    p["tmin"], p["tmax"], p["fs"], p["freq_wav"], p["mu"], p["sigma"]
)
stim = random_pulses(  # stimulus feature
    p["dur"], p["fs"], p["rate"], p["wmin"], p["wmax"], p["amin"], p["amax"]
)
t = np.linspace(0, len(stim) / p["fs"], len(stim))

conf = jitter_pulses(stim, 3)
resp = np.convolve(stim, tf, mode="full")[: len(stim)]

"""
plt.plot(t, stim)
plt.plot(t, conf)
plt.xlim(0, 10)
plt.title(f"r={pearsonr(stim, conf).round(3)}")
plt.show()
"""

reg = np.logspace(-5, 2, 10)
for snr_db in [-10, -20, -30, -40]:
    for i_n, noise in enumerate(
        [
            power_law_noise(p["dur"], p["fs"], 0),
            power_law_noise(p["dur"], p["fs"], 2),
        ]
    ):
        noise = scale_noise(resp, noise, snr_db)
        fig, ax = plt.subplots(1, 2)
        trf = TRF(metric=pearsonr)
        trf.train(
            np.array_split(stim, 5),
            np.array_split(resp + noise, 5),
            p["fs"],
            p["tmin"],
            p["tmax"],
            reg,
        )
        ax[0].plot(trf.times, trf.weights.flatten(), label="stimulus")
        trf.train(
            np.array_split(conf, 5),
            np.array_split(resp + noise, 5),
            p["fs"],
            p["tmin"],
            p["tmax"],
            reg,
        )
        ax[0].plot(trf.times, trf.weights.flatten(), label="confound")
        trf.train(
            np.array_split(np.stack([stim, conf], axis=1), 5),
            np.array_split(resp + noise, 5),
            p["fs"],
            p["tmin"],
            p["tmax"],
            reg,
        )

        for i, (w, l) in enumerate(zip(trf.weights, ["stimulus", "confound"])):
            ax[1].plot(trf.times, w.flatten(), label=l)
        ax[1].legend()
        plt.show()
