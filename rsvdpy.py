from util import clamp


def train(ratings, U, V, lrate=0.001, lamb=0.02):
    for i, j, truth in ratings:
        pred = clamp(U[i].dot(V[:, j]), 1, 5)

        err = truth - pred

        U[i] += lrate * (err * V[:, j] - lamb * U[i])
        V[:, j] += lrate * (err * U[i] - lamb * V[:, j])
    return U, V
