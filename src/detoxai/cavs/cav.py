"""
Credit: https://github.com/frederikpahde/rrclarc
"""

import logging

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


def compute_cav(vecs: np.ndarray, targets: np.ndarray, cav_type: str = "svm") -> tuple:
    """Compute a concept activation vector (CAV) for a set of vectors and targets.

    Args:
      vecs: torch.Tensor of shape (n_samples, n_features)
      targets: torch.Tensor of shape (n_samples,)
      cav_type: str, type of CAV to compute. One of ["svm", "ridge", "signal", "mean"]
      vecs: np.ndarray:
      targets: np.ndarray:
      cav_type: str:  (Default value = "svm")

    Returns:
      torch.Tensor of shape (1, n_features)

    """

    num_targets = (targets == 1).sum()
    num_notargets = (targets == 0).sum()
    weights = (targets == 1) * 1 / num_targets + (targets == 0) * 1 / num_notargets
    weights = weights / weights.max()

    X = vecs

    if "svm" in cav_type:
        linear = LinearSVC(random_state=0, fit_intercept=True, max_iter=200)
        grid_search = GridSearchCV(
            linear, param_grid={"C": [10**i for i in range(-5, 5)]}
        )
        grid_search.fit(X, targets, sample_weight=weights)
        linear = grid_search.best_estimator_
        logger.debug("Best C:", linear.C)
        # linear.fit(X, targets, sample_weight=weights)
        w = torch.Tensor(linear.coef_)
    elif "ridge" in cav_type:
        clf = Ridge(alpha=100, fit_intercept=True, random_state=0)
        grid_search = GridSearchCV(
            clf, param_grid={"alpha": [10**i for i in range(-5, 5)]}
        )
        grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
        clf = grid_search.best_estimator_
        logger.debug("Best alpha:", clf.alpha)
        # clf.fit(X, targets * 2 - 1, sample_weight=weights)
        w = torch.tensor(clf.coef_)[None]

    elif "lasso" in cav_type:
        from sklearn.linear_model import Lasso

        clf = Lasso(alpha=0.1, fit_intercept=True, max_iter=200, random_state=0)

        grid_search = GridSearchCV(
            clf, param_grid={"alpha": [10**i for i in range(-5, 5)]}
        )
        grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
        clf = grid_search.best_estimator_
        logger.debug("Best alpha:", clf.alpha)
        # clf.fit(X, targets * 2 - 1, sample_weight=weights)
        w = torch.tensor(clf.coef_)[None]

    elif "logistic" in cav_type:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(fit_intercept=True, random_state=0)

        grid_search = GridSearchCV(clf, param_grid={"C": [10**i for i in range(-5, 5)]})
        grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
        clf = grid_search.best_estimator_
        logger.debug("Best C:", clf.C)

        # clf.fit(X, targets, sample_weight=weights)
        w = torch.tensor(clf.coef_)

    elif "signal" in cav_type:
        logger.debug("Calculating signal CAV")
        y = targets
        mean_y = y.mean()
        X_residuals = X - X.mean(axis=0)[None]
        covar = (X_residuals * (y - mean_y)[:, np.newaxis]).sum(axis=0) / (
            y.shape[0] - 1
        )
        vary = np.sum((y - mean_y) ** 2, axis=0) / (y.shape[0] - 1)
        w = covar / vary
        w = torch.tensor(w)[None]
    elif "mean-mass" in cav_type:
        mean_act_nonartif = torch.tensor(X[targets == 0].mean(0), dtype=torch.float32)
        mean_act_artif = torch.tensor(X[targets == 1].mean(0), dtype=torch.float32)

        cav = mean_act_nonartif - mean_act_artif

    else:
        raise NotImplementedError()

    cav = w / torch.sqrt((w**2).sum())
    cav = cav.detach().cpu()

    mean_act_nonartif = torch.tensor(X[targets == 0].mean(0), dtype=torch.float32)
    mean_act_artif = torch.tensor(X[targets == 1].mean(0), dtype=torch.float32)

    # print("largest CAV values:", torch.topk(cav.flatten(), 10))

    return cav, mean_act_nonartif, mean_act_artif
