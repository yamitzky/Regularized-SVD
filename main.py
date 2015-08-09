import pyximport; pyximport.install()

import rsvd
import rsvdpy
from util import rmse, load
from baseline import (AvgPredictor, RandomPredictor,
                      BetterAvgPredictor, SvdPredictor, SvdAvgPredictor)


if __name__ == "__main__":
    ratings = load('u1.base')
    ratings_test = load('u1.test')

    # baseline
    gen_predictor = (
        lambda: RandomPredictor(),  # for memory-efficiency
        lambda: AvgPredictor(),
        lambda: BetterAvgPredictor(),
        lambda: SvdPredictor(K=100),
        lambda: SvdAvgPredictor(K=100),
        lambda : rsvd.RSVD(),
    )

    for gen in gen_predictor:
        model = gen()
        model.fit(ratings)
        err = rmse([r - model.predict(uid, iid)
                    for uid, iid, r in ratings_test])
        print model.__class__.__name__, err

