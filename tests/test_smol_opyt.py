"""Tests for the smol_opyt.my_module module.
"""
import pytest

import numpy as np
from numpy import linalg as LA
from smol_opyt.logistic_problem import LogisticProblem
from smol_opyt.metrics import accuracy

def test_logistic_regression():
    X = np.array(
        [
            [0.0], [0.2], [0.4], [0.6], [0.8], [1.0]
        ]
    )
    y = np.array(
        [0, 0, 1, 0, 1, 1]
    )
    problem = LogisticProblem(X, y)
    yhat = problem.predict()
    ce = np.log(0.5) * len(y)
    assert ce == problem.cross_entropy(yhat)

    g1 = problem.cross_entropy_gradient(yhat)
    g2 = np.zeros(2)
    for i in range(0, 6):
        g2 = g2 + (y[i]-yhat[i]) * np.array([1.0, *X[i]])
    assert LA.norm(g1 - g2) < 1e-12

    problem._beta[0] = -1.0
    problem._beta[1] = 4.0
    he = problem.sigmoid(np.array([-1.0, -0.2, 0.6, 1.4, 2.2, 3.0]))
    ce = sum([y[i] * np.log(he[i]) + (1 - y[i]) * np.log(1 - he[i]) for i in range(0,6)])
    yhat = problem.predict(X)
    assert ce == problem.cross_entropy(yhat)

    g1 = problem.cross_entropy_gradient(yhat)
    g2 = np.zeros(2)
    for i in range(0, 6):
        g2 = g2 + (y[i]-yhat[i]) * np.array([1.0, *X[i]])
    assert LA.norm(g1 - g2) < 1e-12

    problem.solve()
    print(problem._beta)
    print(problem.predict())
    assert accuracy(y, np.round(problem.predict())) == 2 / 3

def test_metrics():
    assert accuracy(np.array([0, 0, 0]), np.array([1, 1, 1])) == 0.0
    assert accuracy(np.array([0, 0, 1]), np.array([1, 1, 1])) == 1/3
    assert accuracy(np.array([0, 1, 1]), np.array([1, 1, 1])) == 2/3
    assert accuracy(np.array([1, 1, 1]), np.array([1, 1, 1])) == 1.0

