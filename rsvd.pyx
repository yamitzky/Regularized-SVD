# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
cimport cython

DOUBLE = np.float
ctypedef np.float_t DOUBLE_T

@cython.boundscheck(False)
def train(np.ndarray[DOUBLE_T, ndim=2] X,
          np.ndarray[DOUBLE_T, ndim=2] U,
          np.ndarray[DOUBLE_T, ndim=2] V,
          double lrate=0.001, double lamb=0.02):
    cdef int i, j, k
    cdef int n_user = X.shape[0]
    cdef int n_item = X.shape[1]
    cdef int n_topic = U.shape[1]
    cdef DOUBLE_T truth, pred, err

    for i in range(n_user):
        for j in range(n_item):
            if X[i, j] == 0:
                continue

            truth = X[i, j]
            pred = 0.0
            for k in range(n_topic):
                pred += U[i, k] * V[k, j]
            if pred > 5:
                pred = 5
            elif pred < 1:
                pred = 1
            err = truth - pred

            for k in range(n_topic):
                U[i, k] += lrate * (err * V[k, j] - lamb * U[i, k])
                V[k, j] += lrate * (err * U[i, k] - lamb * V[k, j])
    # notice: 違うっぽい
