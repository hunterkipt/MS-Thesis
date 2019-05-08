#!/usr/bin/env python3
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import os
import prediction_error as pe
from statsmodels.tsa.ar_model import AR

def feature_extract(perror_seq, ar=False, model_order=4):
# Takes in a prediction error sequence and returns a tuple containing three
# Features: (0) The energy in the fingerprint s^(n), (1) the average (mean) of
# the prediction error signal, and (2) the variance of the prediction error
# signal.
    # First get e^(n), the median filtered perror sequence.
    e_hat = sig.medfilt(perror_seq, kernel_size=3)
    zero = np.zeros(perror_seq.shape)
    # s^(n) = max(perror_seq - e^(n), 0)
    s_hat = np.maximum(perror_seq - e_hat, zero)
    if not ar:
        # Sum over all elements of s^(n) and divide by signal length
        # To get signal energy
        energy = np.sum(np.abs(s_hat))/len(s_hat)
        avg_shat = np.mean(s_hat)
        var_shat = np.var(s_hat, dtype=np.float64)
        avg_en = np.mean(perror_seq)
        var_en = np.var(perror_seq, dtype=np.float64)

        return (energy, avg_shat, var_shat, avg_en, var_en)

    else:
        # Fit an AR model to the prediction error sequence
        model = AR(s_hat)
        model_fitted = model.fit(maxlag=model_order)

        return model_fitted.params

def get_features(input_path, ar=False, model_order=4):
# This function will take in a directory path, and perform feature extraction
# on all files in the directory path. The function will save a file with
# file name "<dir>.txt", with the corresponding feature matrix of all perror
# sequences in the directory. The function will also return a numpy array
# containing these features.
    if not os.path.isdir(input_path):
        return None
    
    folder_name = input_path.split("/")[-1]
    f_matrix = []
    for filename in sorted(os.listdir(input_path)):
        file_path = os.path.join(input_path, filename)
        if not os.path.isfile(file_path):
            continue

        # Load perror sequence from file
        try:
            perror_seq = pe.load_perror_sequence(file_path)
            if ar and perror_seq.shape[0] < model_order + 2:
                continue

        except:
            # Can't extract features. Skip file
            continue

        # only the third column has the values of the perror. Extract features
        features = feature_extract(perror_seq[:,2], ar, model_order)
        f_matrix.append(features)
    
    if len(f_matrix) == 0:
        return None
    # All Features extracted. Save and return the array
    
    f_matrix = np.array(f_matrix)
    if ar:
        header = folder_name + "\nAR Model Fit"
        output_path = os.path.join(input_path, folder_name + "_AR.txt")

    else:
        header = folder_name + "\nenergy, avg_s_hat, var_s_hat, avg_en, var_en"
        output_path = os.path.join(input_path, folder_name + ".txt")
    
    np.savetxt(output_path, f_matrix, fmt="%.10f", delimiter=',', header=header)

    return f_matrix

def construct_ROC(clean_features, fdel_features):
    if clean_features is None or fdel_features is None:
        return None

    clean_energy = clean_features.T[0]
    fdel_energy = fdel_features.T[0]
    max_thresh = np.maximum(np.amax(clean_energy), np.amax(fdel_energy))
    # Test the features at a significant number of thresholds
    threshes = np.linspace(0, max_thresh, 150)
    ROC = []
    for thresh in threshes:
        pfa = (clean_energy > thresh).sum()/len(clean_energy)
        pd = (fdel_energy > thresh).sum()/len(fdel_energy)
        ROC.append((thresh, pfa, pd))

    ROC = np.array(ROC)
    return ROC

def plot_ROC(clean_data_path, fdel_data_path):
    clean_features = get_features(clean_data_path)
    fdel_features = get_features(fdel_data_path)
    ROC = construct_ROC(clean_features, fdel_features)
    
    clean_data_folder = clean_data_path.split("/")[-1]
    plt.plot(ROC[:,1], ROC[:,2])
    plt.title(clean_data_path + "ROC Curve")
    plt.xlabel("P(FA)")
    plt.ylabel("P(D)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(clean_data_path + "/ROC.png", dpi=300, format="png")

def plot_feature_hist(clean_data_path, fdel_data_path):
    clean_features = get_features(clean_data_path)
    fdel_features = get_features(fdel_data_path)
    
    clean_energy = clean_features.T[0]
    fdel_energy = fdel_features.T[0]
    max_thresh = np.maximum(np.amax(clean_energy), np.amax(fdel_energy))
    bins = np.linspace(0, max_thresh, 25)
    plt.hist(clean_energy, bins=bins, density=True, label="Clean Data")
    plt.hist(fdel_energy, bins=bins, density=True, label="Frame Deleted")
    plt.title(clean_data_path + " Perror Energy Histogram")
    plt.xlabel("Perror Signal Energy")
    plt.legend()
    plt.savefig(clean_data_path + "/HIST.png", dpi=300, format="png")

def load_dataset(in_file):
    '''
    Loads a prediction error feature dataset from a file and returns
    it as a numpy array.
    in_file = path to the input file where your dataset is
    '''
    converters = {0: lambda s: float(s.strip() or 0),
                  1: lambda s: float(s.strip() or 0),
                  2: lambda s: float(s.strip() or 0),
                  3: lambda s: float(s.strip() or 0),
                  4: lambda s: float(s.strip() or 0)}
    data_set = np.loadtxt(in_file, delimiter=',', converters=converters)
    return data_set 
