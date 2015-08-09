from itertools import groupby
from operator import itemgetter

import numpy as np
import scipy as sp
import scipy.sparse

def to_sparse(ratings, shape=None):
    _1, _2, _3 = itemgetter(0), itemgetter(1), itemgetter(2)
    data = map(_3, ratings)
    i = map(_1, ratings)
    j = map(_2, ratings)
    return sp.sparse.coo_matrix((data, (i, j)), shape=shape).tocsr()


def sgroupby(iterable, key):
    return groupby(sorted(iterable, key=key), key)


def clamp(num, min_, max_):
    return min(max(num, min_), max_)


def rmse(residuals):
    return np.sqrt(np.average(np.square(residuals)))


def load(name):
    with open('data/' + name) as f:
        data = []
        for line in f:
            line = line.strip().split('\t')
            data.append((
                int(line[0]) - 1,
                int(line[1]) - 1,
                float(line[2])
            ))
        return data

