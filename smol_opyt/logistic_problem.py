from math import log
import numpy as np
from numpy import linalg as LA

class LogisticProblem:
    """Class for the logistic regression method for classification."""

    def __init__(self, feat_mtx, y):
        """Create a Logistic Problem with matrix `feat_mtx` n by p and vector `y` of 0s and 1s with size n. A bias is added to the model as the first variable."""
        self._feat_mtx = feat_mtx
        self._y = y
        (n, p) = feat_mtx.shape
        self._beta = np.zeros(p + 1)

    def sigmoid(self, v):
        """Compute sigmoid(v) = 1 / (1 + exp(-v)"""
        return 1 / (1 + np.exp(-v))

    def predict(self, feat_mtx=None, beta=None):
        if feat_mtx is None:
            feat_mtx = self._feat_mtx
        if beta is None:
            beta = self._beta
        return self.sigmoid(beta[0] + np.dot(feat_mtx, beta[1:]))

    def cross_entropy(self, yhat):
        """Compute the cross entropy, given by

            sum y[i] * log(yhat[i]) + (1 - y[i]) * log(1 - yhat[i])"""
        n = len(self._y)
        c = 0.0
        for i in range(0, n):
            c += self._y[i] * log(yhat[i]) + (1 - self._y[i]) * log(1 - yhat[i])

        return c

    def cross_entropy_gradient(self, yhat):
        """Assuming yhat_i = sigmoid(x_i^T beta), returns

            sum (y[i] - yhat) * x_i
        """
        n = len(self._y)
        p = len(self._beta)
        g = np.zeros(p)
        for i in range(0, n):
            g = g + (self._y[i] - yhat[i]) * np.array([1.0, *self._feat_mtx[i,:]])

        return g

    def solve(self):
        """Solve the logistic regression problem"""
        max_iter = 1000
        iter = 0
        yhat = self.predict()
        L = self.cross_entropy(yhat)
        gradL = self.cross_entropy_gradient(yhat)
        eta = 0.01
        while LA.norm(gradL) > 1e-6 and iter < max_iter:
            alpha = 1.0
            slope = LA.norm(gradL)**2
            beta_new = self._beta + alpha * gradL
            yhat = self.predict(beta=beta_new)
            L_new = self.cross_entropy(yhat)
            while L_new < L + 1e-4 * alpha * slope:
                alpha = alpha / 2
                beta_new = self._beta + alpha * gradL
                yhat = self.predict(beta=beta_new)
                L_new = self.cross_entropy(yhat)
            self._beta = beta_new
            L = L_new
            gradL = self.cross_entropy_gradient(yhat)
            iter += 1