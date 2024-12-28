import torch
import numpy as np


def balanced_accuracy_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate the balanced accuracy metric
    """

    y_true = y_true.int()
    y_pred = y_pred.int()

    # Compute confusion matrix
    n_classes = len(torch.unique(y_true))
    confusion_matrix = torch.zeros(n_classes, n_classes)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    # Compute balanced accuracy
    balanced_acc = 0
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, :].sum() - tp
        fp = confusion_matrix[:, i].sum() - tp
        tn = confusion_matrix.sum() - tp - fn - fp
        balanced_acc += tp / (tp + fn) + tn / (tn + fp)
    balanced_acc /= 2 * n_classes

    return balanced_acc


def balanced_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the balanced accuracy metric
    """

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Compute confusion matrix
    n_classes = len(np.unique(y_true))
    confusion_matrix = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    # Compute balanced accuracy
    balanced_acc = 0
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, :].sum() - tp
        fp = confusion_matrix[:, i].sum() - tp
        tn = confusion_matrix.sum() - tp - fn - fp
        balanced_acc += tp / (tp + fn) + tn / (tn + fp)
    balanced_acc /= 2 * n_classes

    return balanced_acc
