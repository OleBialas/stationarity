from pathlib import Path
import json
import numpy as np
from slab import Sound, Filter

root = Path(__file__).parent.parent.absolute()
p = json.load(open(root / "code" / "parameters.json"))
np.random.seed(p["seed"])

wav_files = list((root / "data" / "stimuli").glob("*.wav"))
wav_files.sort()
for i_b, bands in enumerate(p["stim"]["n_bands"]):
    print(f"Computing spectrogram with {bands} bands ...")
    for w in wav_files:
        fname = w.name.split(".")[0]
        sound = Sound(w).channel(0)
        sound = sound.filter(
            frequency=(p["stim"]["low_cutoff"], p["stim"]["high_cutoff"]),
            kind="bp",
        )
        sound = sound.resample(int(sound.samplerate / 2))  # downsample for performance
        fbank = Filter.cos_filterbank(samplerate=sound.samplerate, n_filters=bands)
        subbands = fbank.apply(sound)
        spectrogram = subbands.envelope(cutoff=p["stim"]["env_cutoff"])
        spectrogram = spectrogram.resample(p["fs"])
        spectrogram = spectrogram.data
        spectrogram = spectrogram.clip(min=0)
        np.save(
            root / "results" / "spectrogram" / f"{fname}_spg_{bands}bands.npy",
            spectrogram,
        )
