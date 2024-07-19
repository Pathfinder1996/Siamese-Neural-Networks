import numpy as np

from sklearn.metrics import auc, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

def compute_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean([fpr[min_index], fnr[min_index]])
    return eer, thresholds[min_index]

def plot_det_curve(fpr, fnr, eer, label=None):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, fnr, label=label)
    eer_index = np.nanargmin(np.absolute(fpr - fnr))
    eer_fpr = fpr[eer_index]
    eer_fnr = fnr[eer_index]
    plt.scatter(eer_fpr, eer_fnr, color="black", label=f"EER = {eer:.5f}%")
    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate(%)")
    plt.ylabel("False Negative Rate(%)")
    plt.title("DET Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def calculate_confusion_matrix(labels, predictions, threshold):
    pred_labels = (predictions > threshold).astype(int)

    conf_matrix = confusion_matrix(labels, pred_labels)
    return conf_matrix

def plot_confusion_matrix(conf_matrix, classes, title="Confusion matrix", cmap=plt.cm.Blues):
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

def plot_density(genuine_scores, impostor_scores):
    sns.kdeplot(genuine_scores, bw_adjust=2, label="Genuine")
    sns.kdeplot(impostor_scores, bw_adjust=2, label="Impostor")
    plt.title("Probability Density")
    plt.xlabel("Distance")
    plt.legend()
    plt.show()