#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import perror_features as pf

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from matplotlib.font_manager import FontProperties

def plot_svm(clean_ds, fdel_ds, ax, label_svm):
    print("Formatting data")
    # Make label vectors (0 is clean, 1 is frame_deleted)
    clean_labels = np.zeros((clean_ds.shape[0], 1))
    fdel_labels = np.ones((fdel_ds.shape[0], 1))

    # Make X, the dataset, and y the label vector by stacking the imported data
    X = np.vstack((clean_ds, fdel_ds))
    y = np.vstack((clean_labels, fdel_labels))

    # Binarize the output
    y = label_binarize(y, classes=[0, 1])
    n_classes = y.shape[1]

    print("Shuffling and splitting data")
    # Shuffle and split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,
                                                        random_state=0)

    print("Fitting Classifier")
    # Fit a classifier
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                     random_state=0))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    print("Computing ROC curves")
    # Compute ROC curve and area
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve vs random chance
    
    lw = 2
    ax.plot(fpr[0], tpr[0], lw=lw, 
            label=(label_svm + ' (AUC = {:0.2f})'.format(roc_auc[0])))
    ax.plot([0, 1], [0, 1], lw = lw, linestyle='--', color='g')

def plot_energy(clean_ds, fdel_ds, ax, label_energy):
    energy_ROC = pf.construct_ROC(clean_ds, fdel_ds)
    energy_auc = auc(energy_ROC[:, 1], energy_ROC[:, 2])

    lw = 2
    ax.plot(energy_ROC[:, 1], energy_ROC[:, 2], lw=lw,
            label=(label_energy + ' (AUC = {:0.2f})'.format(energy_auc)))

print("Importing dataset")
# Import Clean and Frame Deleted Datasets
clean_ds_path = "../videos/KodakEktra/perror_infer_new"
fdel_ds_path = "../videos/KodakEktra_fdel/perror_infer_new"

fig, ax = plt.subplots()

for i in range(4, 30, 10):
    clean_ds = pf.get_features(clean_ds_path, ar=True, model_order=i)
    fdel_ds = pf.get_features(fdel_ds_path, ar=True, model_order=i)

    label_svm = f'AR Model SVM [Model Order = {i}]'
    #label_energy = 'Baseline detector [Kodak Ektra]'
    plot_svm(clean_ds, fdel_ds, ax, label_svm)
    #plot_energy(clean_ds, fdel_ds, ax, label_energy)

clean_ds_path = "../videos/KodakEktra/perror_infer_new"
fdel_ds_path = "../videos/KodakEktra_fdel/perror_infer_new"

clean_ds = pf.get_features(clean_ds_path)
fdel_ds = pf.get_features(fdel_ds_path)

label_svm = 'New features + SVM [Asus ZenFone 3]'
#label_energy = 'Baseline detector [Asus ZenFone 3]'
plot_svm(clean_ds, fdel_ds, ax, label_svm)
#plot_energy(clean_ds, fdel_ds, ax, label_energy)

fontP = FontProperties()
fontP.set_size('small')
ax.set_xlim([0.0, 1.05])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel(r'$P_{FA}$')
ax.set_ylabel(r'$P_{D}$')
ax.set_title('Frame Deletion Detection ROC')
box = ax.get_position()
ax.set_position([box.x0 + box.width * 0.2, box.y0 + box.height * 0.2,
                 box.width * 0.8, box.height * 0.8])

# Put a legend below current axis

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, prop=fontP)
plt.savefig("./perror_AR_model_comparison_roc.png", dpi=300, format="png")
