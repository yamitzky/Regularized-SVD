from operator import itemgetter

import scipy as sp
import scipy.sparse

def to_sparse(ratings, shape=None):
    _1, _2, _3 = itemgetter(0), itemgetter(1), itemgetter(2)
    data = map(_3, ratings)
    i = map(_1, ratings)
    j = map(_2, ratings)
    return sp.sparse.coo_matrix((data, (i, j)), shape=shape).tocsr()
