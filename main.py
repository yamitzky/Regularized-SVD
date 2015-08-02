import numpy as np

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
