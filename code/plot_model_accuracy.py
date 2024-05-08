from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt

plt.style.use("science")
p = json.load(open(root / "code" / "effect_of_segmentation_parameters.json"))
root = Path(__file__).parent.parent.absolute

dur_segments = [1, 10, 30, 60, 120]
n_bands = [1, 16, 32]
for sub in (root / "results" / "fit").glob("*_accuracy.npy"):
    acc = np.load(sub)
