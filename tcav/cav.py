import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC


def compute_cav(vecs: np.ndarray, targets: np.ndarray, cav_type: str = "svm"):
    """
    Compute a concept activation vector (CAV) for a set of vectors and targets.

    :param vecs:    torch.Tensor of shape (n_samples, n_features)
    :param targets: torch.Tensor of shape (n_samples,)
    :param cav_type:   str, type of CAV to compute. One of ["svm", "ridge", "signal", "mean"]
    :return:       torch.Tensor of shape (1, n_features)
    """

    num_targets = (targets == 1).sum()
    num_notargets = (targets == 0).sum()
    weights = (targets == 1) * 1 / num_targets + (targets == 0) * 1 / num_notargets
    weights = weights / weights.max()

    X = vecs

    if "svm" in cav_type:
        linear = LinearSVC(random_state=0, fit_intercept=False)
        grid_search = GridSearchCV(linear, param_grid={"C": [10 ** i for i in range(-5, 5)]})
        grid_search.fit(X, targets, sample_weight=weights)
        linear = grid_search.best_estimator_
        print("Best C:", linear.C)
        # linear.fit(X, targets, sample_weight=weights)
        w = torch.Tensor(linear.coef_)
    elif "ridge" in cav_type:

        clf = Ridge(alpha=100, fit_intercept=False)
        grid_search = GridSearchCV(clf, param_grid={"alpha": [10 ** i for i in range(-5, 5)]})
        grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
        clf = grid_search.best_estimator_
        print("Best alpha:", clf.alpha)
        # clf.fit(X, targets * 2 - 1, sample_weight=weights)
        w = torch.tensor(clf.coef_)[None]

    elif "lasso" in cav_type:
        from sklearn.linear_model import Lasso
        clf = Lasso(alpha=0.01, fit_intercept=False)

        grid_search = GridSearchCV(clf, param_grid={"alpha": [10 ** i for i in range(-5, 5)]})
        grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
        clf = grid_search.best_estimator_
        print("Best alpha:", clf.alpha)
        # clf.fit(X, targets * 2 - 1, sample_weight=weights)
        w = torch.tensor(clf.coef_)[None]

    elif "logistic" in cav_type:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(fit_intercept=False)

        grid_search = GridSearchCV(clf, param_grid={"C": [10 ** i for i in range(-5, 5)]})
        grid_search.fit(X, targets * 2 - 1, sample_weight=weights)
        clf = grid_search.best_estimator_
        print("Best C:", clf.C)

        # clf.fit(X, targets, sample_weight=weights)
        w = torch.tensor(clf.coef_)

    elif "signal" in cav_type:
        print("SIGNAL RIDGE")
        y = targets
        mean_y = y.mean()
        X_residuals = X - X.mean(axis=0)[None]
        covar = (X_residuals * (y - mean_y)[:, np.newaxis]).sum(axis=0) / (y.shape[0] - 1)
        vary = np.sum((y - mean_y) ** 2, axis=0) / (y.shape[0] - 1)
        w = (covar / vary)
        w = torch.tensor(w)[None]

    else:
        raise NotImplementedError()


    cav = w / torch.sqrt((w ** 2).sum())
    cav = cav.detach().cpu()

    mean_activation_over_nonartifact_samples = X[targets == 0].mean(0)
    print("CAV type: ", cav_type)
    print("largest CAV values:", torch.topk(cav.flatten(), 10))
    mean_activation_over_nonartifact_samples_tensor = torch.tensor(mean_activation_over_nonartifact_samples)
    return cav, mean_activation_over_nonartifact_samples_tensor
