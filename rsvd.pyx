# -*- coding: utf-8 -*-
from itertools import imap

import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from sklearn.utils import shuffle

from cpprand cimport uniform_int_distribution, mt19937, random_device
from util import rmse


ctypedef uniform_int_distribution[int] int_dist


class RSVD(object):
    def __init__(self, K=100, lrate=0.015, lamb=0.05):
        self.K = K
        self.lrate = lrate
        self.lamb = lamb

    def fit(self, ratings, max_iter=10000, seed=None):
        cdef int c
        cdef int rank = self.K
        cdef int n_users, n_items
        cdef int n_ratings = len(ratings)
        cdef int s
        cdef np.ndarray[double, ndim=2] U, V, X
        cdef random_device *rand_gen

        assert np.array(ratings).shape[1] == 3, 'ratings must be 3-column array'

        n_users = max(imap(lambda x: x[0], ratings)) + 1
        n_items = max(imap(lambda x: x[1], ratings)) + 1

        if seed is None:
            rand_gen = new random_device()
            s = rand_gen[0]()
        else:
            s = <int>seed

        # if already trained, re-train parameters. otherwise, initilize with random values
        U = self.__dict__.get("U_", np.random.uniform(-1.0/rank, 1.0/rank, size=(n_users, rank)))
        V = self.__dict__.get("V_", np.random.uniform(-1.0/rank, 1.0/rank, size=(rank, n_items)))

        X = np.array(ratings)
        train(X, U, V, n_ratings, self.K, self.lrate, self.lamb,
              max_iter, s)

        self.U_ = U
        self.V_ = V

    def predict(self, int user_id, int item_id):
        return predict(self.U_, self.V_, user_id, item_id, self.K)


@cython.boundscheck(False)
cdef void train(double[:, :] ratings,
                double[:, :] U,
                double[:, :] V,
                int n_ratings, int n_topic,
                double lrate, double lamb,
                int max_iter, int seed):
    cdef int i, j, k, t, c
    cdef double truth, err, u, v
    cdef mt19937 *engine = new mt19937(seed)
    cdef int_dist *randint = new int_dist(0, n_ratings - 1)

    for c in range(max_iter):
        t = randint[0](engine[0])
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
