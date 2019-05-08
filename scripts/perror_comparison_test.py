#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from prediction_error import load_perror_sequence
import os
from scipy.signal import medfilt

vids = ["V_19700108_113324_vHDR_Auto.txt", "V_19700108_113256_vHDR_Auto.txt"]

dirs = ["../videos/ASUSZenFone3/perror", "../videos/ASUSZenFone3/perror_new",
        "../videos/ASUSZenFone3/perror_ds"]

for vid in vids:
    perror_seqs = [load_perror_sequence(os.path.join(d, vid)) for d in dirs]
    perror_seqs[1][:, 2] *= (np.mean(perror_seqs[0][:, 2])/np.mean(perror_seqs[1][:, 2]))
    perror_seqs[1][:, 2] = medfilt(perror_seqs[1][:, 2])
    fig, ax = plt.subplots()
    [ax.plot(seq[:, 0], seq[:, 2]) for seq in perror_seqs]
    ax.set_title(vid + " Prediction error sequence comparison")
    ax.set_xlabel("P-Frame Index")
    ax.set_ylabel("Prediction Error")
    ax.legend(["Old FFMPEG Method", "New FFMPEG Method", "Diamond Search Method"])
    plt.show()



