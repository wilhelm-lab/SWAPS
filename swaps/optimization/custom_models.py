import logging
from typing import Literal, Union

import numpy as np
from scipy.optimize import minimize

Logger = logging.getLogger(__name__)


class CustomLinearModel:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L1 regularization
    """

    def __init__(
        self,
        X,
        Y,
        sample_weights=None,
        residue_loss=None,
        beta_init=None,
        reg_norm: Union[Literal["l1", "l2"], None] = None,
        reg_param: float = 0.0,
    ):
        self.X = X
        self.Y = Y
        self.sample_weights = sample_weights
        self.beta_init = beta_init
        self.beta = None

        self.residue_loss = residue_loss

        self.regularization_norm = reg_norm
        self.regularization_strength = reg_param

    def predict(self, X):
        prediction = np.matmul(X, self.beta)
        return prediction

    def model_error(self):
        error = self.residue_loss(self.predict(self.X), self.Y, self.sample_weights)
        return error

    def total_loss(self, beta):
        self.beta = beta
        match self.regularization_norm:
            case None:
                return self.model_error()
            case "l1":
                return self.model_error() + sum(
                    self.regularization_strength * np.array(self.beta)
                )
            case "l2":
                return self.model_error() + sum(
                    self.regularization_strength * np.array(self.beta) ** 2
                )

    def fit(self, maxiter=250, method: str = "TNC"):
        if self.beta_init is None:
            self.beta_init = np.array([1] * self.X.shape[1])  # default init: beta = 1
        else:
            pass

        if self.beta is not None and all(self.beta_init == self.beta):
            Logger.info("Model already fit once; continuing fit with more itrations.")

        # option_key = "maxfun" if method == "TNC" else "maxiter"
        res = minimize(
            self.total_loss,
            self.beta_init,
            method=method,
            tol=0.001,
            # options={option_key: maxiter},
            bounds=[(0, None) for _ in range(self.X.shape[1])],
        )  # only positive values
        self.beta = res.x
        self.beta_init = self.beta


def mean_square_root_error(y_pred, y_true, sample_weights=None):
    assert y_pred.all() >= 0 & y_true.all() >= 0
    y_true_sqrt = y_true
    y_pred_sqrt = y_pred
    assert len(y_true_sqrt) == len(y_pred_sqrt)
    if sample_weights is None:
        return np.mean(np.square(y_true_sqrt - y_pred_sqrt))
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true_sqrt)
        return np.dot(sample_weights, (np.square(y_true_sqrt - y_pred_sqrt))) / sum(
            sample_weights
        )


def MSRE_loss(beta, X, Y, sample_weights=None):
    return mean_square_root_error(np.matmul(X, beta), Y, sample_weights=sample_weights)
