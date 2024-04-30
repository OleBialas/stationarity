from pathlib import Path
import json
import numpy as np
from mtrf import TRF
from generators import wavelet, random_pulses, power_law_noise, scale_noise
from utils import segment_data

root = Path(__file__).parent.parent.absolute()
p = json.load(open(root / "code" / "pink_vs_white_sim_parameters.json"))
np.random.seed(p["seed"])

# simulate stimuli and neural response
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

acc, reg = [
    np.zeros((len(p["snr_db"]), p["n_reps"], len(p["alpha"]), len(p["dur_segment"])))
    for _ in range(2)
]

for i_r in range(p["n_reps"]):
    print(i_r)
    for i_a, alpha in enumerate(p["alpha"]):  # use white and pink noise
        noise = power_law_noise(p["dur"], p["fs"], alpha)
        for i_s, snr_db in enumerate(p["snr_db"]):
            noise_scaled = scale_noise(resp_train, noise, snr_db)
            noise_scaled = np.expand_dims(noise_scaled, axis=1)
            for i_d, dur in enumerate(p["dur_segment"]):
                stim_train_segments, resp_train_segments = segment_data(
                    [stim_train], [resp_train + noise_scaled], dur, normalize=True
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

                reg[i_r, i_a, i_s, i_d] = trf.regularization
                acc[i_r, i_a, i_s, i_d] = trf.predict(
                    stim_test_segments, resp_test_segments
                )[-1]
np.save(root / "results" / "white_vs_pink_noise_sim_lambda.npy", reg)
np.save(root / "results" / "white_vs_pink_noise_sim_accuracy.npy", acc)
