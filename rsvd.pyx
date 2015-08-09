# -*- coding: utf-8 -*-
from itertools import imap

import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from sklearn.utils import shuffle

from util import rmse


class RSVD(object):
    def __init__(self, K=100, lrate=0.015, lamb=0.05, max_iter=30):
        self.K = K
        self.lrate = lrate
        self.lamb = lamb
        self.max_iter = max_iter

    def fit(self, ratings, ratings_test=None):
        cdef int rank = self.K
        cdef int max_iter = self.max_iter
        cdef int n_users, n_items, n_ratings
        cdef int c
        cdef np.ndarray[double, ndim=2] U, V, X

        assert np.array(ratings).shape[1] == 3, 'ratings must be 3-column array'

        n_users = max(imap(lambda x: x[0], ratings)) + 1
        n_items = max(imap(lambda x: x[1], ratings)) + 1
        n_ratings = len(ratings)
        U = np.random.uniform(-1.0/rank, 1.0/rank, size=(n_users, rank))
        V = np.random.uniform(-1.0/rank, 1.0/rank, size=(rank, n_items))

        for c in range(max_iter):
            X = shuffle(ratings, random_state=c)
            train(X, U, V, n_ratings, self.K, self.lrate, self.lamb)
            if ratings_test:
                err = rmse([r - clamp(U[uid].dot(V[:, iid]), 1, 5)
                            for uid, iid, r in ratings_test])
                print err
        self.U_ = U
        self.V_ = V

    def predict(self, int user_id, int item_id):
        return predict(self.U_, self.V_, user_id, item_id, self.K)


@cython.boundscheck(False)
cdef void train(double[:, :] ratings,
                double[:, :] U,
                double[:, :] V,
                int n_ratings, int n_topic,
                double lrate, double lamb):
    cdef int i, j, k, t
    cdef double truth, err, u, v

    for t in range(n_ratings):
        i = <int>ratings[t, 0]
        j = <int>ratings[t, 1]
        truth = ratings[t, 2]

        err = truth - predict(U, V, i, j, n_topic)

        for k in range(n_topic):
            u = U[i, k]
            v = V[k, j]
            U[i, k] += lrate * (err * v - lamb * u)
            V[k, j] += lrate * (err * u - lamb * v)


cdef inline double clamp(double value, double min_, double max_):
    return max(min(value, max_), min_)


@cython.boundscheck(False)
cdef inline float predict(
    double[:, :] U,
    double[:, :] V,
    int user_id, int item_id, int n_topic):

    cdef double pred = 0.0
    for k in range(n_topic):
        pred += U[user_id, k] * V[k, item_id]

    return clamp(pred, 1.0, 5.0)
