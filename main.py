import numpy as np
from sklearn.decomposition import TruncatedSVD

from dataset import Rating

if __name__ == "__main__":
    ratings = Rating.load("u1.base")
    ratings_test = Rating.load("u1.test")

    svd = TruncatedSVD(n_components=16, algorithm='arpack')
    topic = svd.fit_transform(ratings.mat)
    pred = svd.inverse_transform(topic)
    pred[pred < 1] = 1
    pred[pred > 5] = 5

    np.average([np.abs(r - pred[uid, iid])
                for uid, iid, r in ratings_test.data])
