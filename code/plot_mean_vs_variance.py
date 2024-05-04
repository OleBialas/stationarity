from pathlib import Path
import json
import numpy as np
from generators import power_law_noise, scale_noise
from matplotlib import pyplot as plt
import scienceplots

root = Path(__file__).parent.parent.absolute()
plt.style.use("science")
p = json.load(open(root / "code" / "mean_vs_variance_sim_parameters.json"))
np.random.seed(p["seed"])

acc = np.load(root / "results" / "variance_nonstationary_accuracy.npy")
reg = np.load(root / "results" / "variance_nonstationary_lambda.npy")
reg = np.log10(reg)
n_resample = 1000

for i_b, b in enumerate(p["beta"]):
    plt.plot(
        p["dur_segment"],
        acc[:, i_b, :].mean(axis=0) / acc[:, i_b, :].mean(axis=0).max(),
        label=b,
    )
plt.legend()
plt.show()
