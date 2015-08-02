from operator import itemgetter

import numpy as np
from sklearn.decomposition import TruncatedSVD

from util import sgroupby, to_sparse, clamp

key_u = itemgetter(0)
key_i = itemgetter(1)
key_r = itemgetter(2)


class RandomPredictor(object):
    def fit(self, ratings):
        pass

    def predict(self, user_id, item_id):
        return np.random.uniform(1, 5)


class AvgPredictor(object):
    def fit(self, ratings):
        self.avg_item = {iid: np.mean(map(key_r, data))
                         for iid, data in sgroupby(ratings.data, key_i)}
        self.global_avg_item = np.mean(map(key_r, ratings.data))

        self.offset_user = {
            uid: np.mean([r - self.avg_item[i] for _, i, r in data])
            for uid, data in sgroupby(ratings.data, key_u)}
        self.global_offset_user = np.mean(
            [r - self.avg_item[i] for _, i, r in ratings.data]
        )

    def predict(self, user_id, item_id):
        return clamp(
            self.avg_item.get(item_id, self.global_avg_item) +
            self.offset_user.get(user_id, self.global_offset_user),
            1, 5)


class BetterAvgPredictor(object):
    def fit(self, ratings):
        self.avg_item, self.global_avg_item =\
            self.better_mean(ratings.data, key_i)

        offset_data = [(u, i, r - self.avg_item[i])
                       for u, i, r in ratings.data]
        self.offset_user, self.global_offset_user =\
            self.better_mean(offset_data, key_u)

    @staticmethod
    def better_mean(data, key):
        rs = map(key_r, data)
        global_var = np.var(rs)
        global_avg = np.mean(rs)

        grouped = [(id, map(key_r, d)) for id, d in sgroupby(data, key)]
        K = {id: np.var(d) / global_var for id, d in grouped}
        avg_item = {id: (global_avg * K[id] + np.sum(r)) / (K[id] + len(r))
                    for id, r in grouped}

        return avg_item, global_avg

    def predict(self, user_id, item_id):
        return clamp(
            self.avg_item.get(item_id, self.global_avg_item) +
            self.offset_user.get(user_id, self.global_offset_user),
            1, 5)


class SvdPredictor(BetterAvgPredictor):
    def __init__(self, K):
        self.K = K

    def fit(self, ratings):
        super(SvdPredictor, self).fit(ratings)

        offset = [(u, i, r - self.offset_user[u] - self.avg_item[i])
                  for u, i, r in ratings.data]

        mat = to_sparse(offset,
                        shape=(len(ratings.users), len(ratings.items)))
        svd = TruncatedSVD(n_components=self.K, algorithm='arpack')
        topic = svd.fit_transform(mat)
        self.pred = svd.inverse_transform(topic)

    def predict(self, user_id, item_id):
        return clamp(
            self.pred[user_id, item_id] +
            self.avg_item.get(item_id, self.global_avg_item) +
            self.offset_user.get(user_id, self.global_offset_user),
            1, 5)
