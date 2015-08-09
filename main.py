# -*- coding: utf-8 -*-
from itertools import groupby
from operator import itemgetter

import numpy as np
from scipy.spatial.distance import cdist
import pyximport; pyximport.install()

import rsvd
import rsvdpy
from util import rmse, load
from baseline import (AvgPredictor, RandomPredictor,
                      BetterAvgPredictor, SvdPredictor, SvdAvgPredictor)


def bag_of_categories(categories):
    # categories to be ['cat1|cat2', 'cat3', ...]
    boc = {}
    for cat in categories:
        for c in cat.split('|'):
            boc[c] = boc.get(c, 0) + 1
    return sorted(boc.iteritems(), key=lambda x: x[1])[::-1]



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
        lambda: rsvd.RSVD(),
    )

    for gen in gen_predictor:
        model = gen()
        model.fit(ratings)
        err = rmse([r - model.predict(uid, iid)
                    for uid, iid, r in ratings_test])
        print model.__class__.__name__, err

    with open('data/movies.dat') as f:
        items = [line.strip().split('::') for line in f]
        items = {int(i) - 1: (info, cat) for i, info, cat in items}
    item_category = model.V_.argmax(axis=0)

    # アイテム素性ごとに、どういう映画があるか？
    # 素性ごとに、属する映画数が多い順にソートした上で、カテゴリを表示
    _2 = itemgetter(1)
    cat_hist = [(g, len(list(l))) for g, l in
                groupby(sorted(enumerate(item_category), key=_2), _2)]
    for item_id, cnt in sorted(cat_hist, key=_2):
        print bag_of_categories(
            [items.get(idx, [None, 'Unknown'])[1]
             for idx in np.argwhere(item_category == item_id).T[0]]
        )

    # 似ているアイテムを抽出できるか？
    # 人気のアイテム10個の、類似アイテムを抽出
    rating_hist = [(item_id, len(list(l)))
                      for item_id, l in groupby(sorted(ratings, key=_2), _2)]
    sim = cdist(model.V_.T, model.V_.T, 'euclidean')
    for item_id, _ in sorted(rating_hist, key=_2)[-10:]:
        print "--------------"
        print items[item_id]
        for sim_id in sim[item_id].argsort()[1:6]:
            print "\t" + str(items.get(sim_id))


