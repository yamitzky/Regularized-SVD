import numpy as np
from sklearn.utils import shuffle
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
import rsvd
import rsvdpy

from dataset import Rating
from baseline import (AvgPredictor, RandomPredictor,
                      BetterAvgPredictor, SvdPredictor)


if __name__ == "__main__":
    ratings = Rating.load("u1.base")
    ratings_test = Rating.load("u1.test")

    # baseline
    predictors = (
        RandomPredictor(),
        AvgPredictor(),
        BetterAvgPredictor(),
        SvdPredictor(K=13)
    )

    for model in predictors:
        model.fit(ratings)
        err = np.average([np.abs(r - model.predict(uid, iid))
                          for uid, iid, r in ratings_test.data])
        print model.__class__.__name__, err

    rank = 40
    U = np.ones(shape=(ratings.mat.shape[0], rank)) * 5.0 / rank
    V = np.ones(shape=(rank, ratings.mat.shape[1])) * 5.0 / rank
    for c in range(1000):
        # X = shuffle(ratings.mat).toarray()
        # rsvd.train(X, U, V, lrate=0.01, lamb=0.01)
        X = shuffle(ratings.data)
        U, V = rsvdpy.train(X, U, V, lrate=0.01, lamb=0.005)
        pred = U.dot(V)
        pred[pred < 1] = 1
        pred[pred > 5] = 5
        err = np.average([np.abs(r - pred[uid, iid])
                          for uid, iid, r in ratings_test.data])
        print err
