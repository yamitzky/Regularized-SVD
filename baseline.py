from operator import itemgetter
from itertools import imap

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import svd_flip
from scipy.sparse.linalg import svds

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
                         for iid, data in sgroupby(ratings, key_i)}
        self.global_avg_item = np.mean(map(key_r, ratings))

        self.offset_user = {
            uid: np.mean([r - self.avg_item[i] for _, i, r in data])
            for uid, data in sgroupby(ratings, key_u)}
        self.global_offset_user = np.mean(
            [r - self.avg_item[i] for _, i, r in ratings]
        )

    def predict(self, user_id, item_id):
        return clamp(
            self.avg_item.get(item_id, self.global_avg_item) +
            self.offset_user.get(user_id, self.global_offset_user),
            1, 5)


class BetterAvgPredictor(object):
    def fit(self, ratings):
        self.avg_item, self.global_avg_item =\
            self.better_mean(ratings, key_i)

        offset_data = [(u, i, r - self.avg_item[i])
                       for u, i, r in ratings]
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


class SvdPredictor(object):
    def __init__(self, K):
        self.K = K

    def fit(self, ratings):
        n_users = max(imap(key_u, ratings)) + 1
        n_items = max(imap(key_i, ratings)) + 1

        mat = to_sparse(ratings,
                        shape=(n_users, n_items))
        svd = TruncatedSVD(n_components=self.K, algorithm='arpack')
        self.U = svd.fit_transform(mat)  # U.dot(Sigma)
        self.VT = svd.components_  # VT

    def predict(self, user_id, item_id):
        return clamp(self.U[user_id].dot(self.VT[:, item_id]), 1, 5)


class SvdAvgPredictor(AvgPredictor):
    def __init__(self, K):
        self.K = K

    def fit(self, ratings):
        super(SvdAvgPredictor, self).fit(ratings)

        n_users = max(imap(key_u, ratings)) + 1
        n_items = max(imap(key_i, ratings)) + 1

        offset = [(u, i, r - self.offset_user[u] - self.avg_item[i])
                  for u, i, r in ratings]

        mat = to_sparse(offset,
                        shape=(n_users, n_items))
        svd = TruncatedSVD(n_components=self.K, algorithm='arpack')
        self.U = svd.fit_transform(mat)  # U.dot(Sigma)
        self.VT = svd.components_  # VT

    def predict(self, user_id, item_id):
        pred = self.U[user_id].dot(self.VT[:, item_id])
        return clamp(
            pred +
            self.avg_item.get(item_id, self.global_avg_item) +
            self.offset_user.get(user_id, self.global_offset_user),
            1, 5)
