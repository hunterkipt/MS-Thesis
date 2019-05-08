import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import frame_deletion
import prediction_error

#V_19700116_170516.mp4
#V_19700116_170527.mp4
#V_19700116_170612.mp4
#V_19700116_170642.mp4

orig_vid = "../videos/ASUSZenFone3/V_19700108_113312_vHDR_Auto.mp4"
orig_filename = orig_vid.split("/")[-1].split(".")[0]

new_vid = "./V_19700108_113312_vHDR_Auto_1_15.mp4"
new_filename = new_vid.split("/")[-1].split(".")[0]

#frame_deletion.delete_frames(orig_vid, 1, 15, new_vid)

orig_perror = prediction_error.load_perror_sequence("../videos/ASUSZenFone3/perror_new/V_19700108_113312_vHDR_Auto.txt")
new_perror = prediction_error.load_perror_sequence("../videos/ASUSZenFone3_fdel/perror_new/V_19700108_113312_vHDR_Auto_1_15.txt")

plt.bar(orig_perror[:,0], orig_perror[:,2], width=1)
plt.title(orig_filename + "Prediction Error Sequence")
plt.xlabel("P-Frame Index")
plt.ylabel("Prediction Error")
plt.savefig(orig_filename + "_perror.png", dpi=300, format="png")
plt.clf()

plt.bar(new_perror[:,0], new_perror[:,2], width=1)
plt.title(new_filename + "Frame Deleted Prediction Error Sequence")
plt.xlabel("P-Frame Index")
plt.ylabel("Prediction Error")
plt.savefig(new_filename + "_fdel_perror.png", dpi=300, format="png")
plt.clf()

k1 = np.linspace(-np.pi, np.pi, len(orig_perror[:,2]))
k2 = np.linspace(-np.pi, np.pi, len(new_perror[:,2]))

orig_fft = np.fft.fft(orig_perror[:,2], len(orig_perror[:,2]))
orig_fft = np.fft.fftshift(orig_fft)

new_fft = np.fft.fft(new_perror[:,2], len(new_perror[:,2]))
new_fft = np.fft.fftshift(new_fft)

plt.plot(k1, np.absolute(orig_fft))
plt.title(orig_filename + "FFT")
plt.xlabel("k")
plt.ylabel("|E(k)|")
plt.ylim(0, 200)
plt.savefig(orig_filename + "_fft.png", dpi=300, format="png")
plt.clf()

plt.plot(k2, np.absolute(new_fft))
plt.title(new_filename + "Frame Deleted FFT")
plt.xlabel("k")
plt.ylabel("|E(k)|")
plt.ylim(0, 200)
plt.savefig(new_filename + "fdel_fft.png", dpi=300, format="png")
