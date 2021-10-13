from math import log
import numpy as np
from numpy import linalg as la


class LogisticProblem:
    """Class for the logistic regression method for classification."""
    def __init__(self, feat_mtx, y):
        """Create a Logistic Problem with matrix `feat_mtx` n by p and vector `y` of 0s and 1s with size n.
        A bias is added to the model as the first variable."""
        self._feat_mtx = feat_mtx
        self._y = y
        p = feat_mtx.shape[1]
        self.beta = np.zeros(p + 1)

    def sigmoid(self, v):
        """Compute sigmoid(v) = 1 / (1 + exp(-v)"""
        return 1 / (1 + np.exp(-v))

    def predict(self, feat_mtx=None, beta=None):
        if feat_mtx is None:
            feat_mtx = self._feat_mtx
        if beta is None:
            beta = self.beta
        return self.sigmoid(beta[0] + np.dot(feat_mtx, beta[1:]))

    def cross_entropy(self, yhat):
        """Compute the cross entropy, given by

            sum y[i] * log(yhat[i]) + (1 - y[i]) * log(1 - yhat[i])"""
        n = len(self._y)
        c = 0.0
        for i in range(0, n):
            c += self._y[i] * log(
                yhat[i]) + (1 - self._y[i]) * log(1 - yhat[i])

        return c

    def cross_entropy_gradient(self, yhat):
        """Assuming yhat_i = sigmoid(x_i^T beta), returns

            sum (y[i] - yhat) * x_i
        """
        n = len(self._y)
        p = len(self.beta)
        g = np.zeros(p)
        for i in range(0, n):
            g = g + (self._y[i] - yhat[i]) * np.array(
                [1.0, *self._feat_mtx[i, :]])

        return g

    def solve(self):
        """Solve the logistic regression problem"""
        max_iter = 1000
        iter_count = 0
        yhat = self.predict()
        loss = self.cross_entropy(yhat)
        gradloss = self.cross_entropy_gradient(yhat)
        while la.norm(gradloss) > 1e-6 and iter_count < max_iter:
            alpha = 1.0
            slope = la.norm(gradloss)**2
            beta_new = self.beta + alpha * gradloss
            yhat = self.predict(beta=beta_new)
            loss_new = self.cross_entropy(yhat)
            while loss_new < loss + 1e-4 * alpha * slope:
                alpha = alpha / 2
                beta_new = self.beta + alpha * gradloss
                yhat = self.predict(beta=beta_new)
                loss_new = self.cross_entropy(yhat)
            self.beta = beta_new
            loss = loss_new
            gradloss = self.cross_entropy_gradient(yhat)
            iter_count += 1
