from util import to_sparse


class Rating(object):
    def __init__(self, ratings, users, items):
        self.users = {k - 1: v for k, v in users.iteritems()}
        self.items = {k - 1: v for k, v in items.iteritems()}
        self.data = [(uid - 1, iid - 1, r) for uid, iid, r in ratings]
        self.mat = to_sparse(self.data, shape=(len(users), len(items)))
        pass

    @classmethod
    def load(cls, name):
        users = cls.load_meta("u.user")
        items = cls.load_meta("u.item")
        ratings = cls.load_ratings(name)
        return cls(ratings, users, items)

    @classmethod
    def load_ratings(cls, name):
        with open("data/%s" % name) as f:
            ratings = []
            for line in f:
                user_id, item_id, rating, timestamp = \
                    map(int, line.strip().split("\t"))
                ratings.append((user_id, item_id, float(rating)))

            return ratings

    @classmethod
    def load_meta(cls, name):
        with open("data/%s" % name) as f:
            items = {}
            for line in f:
                line = line.strip().split("|")
                items[int(line[0])] = line[1:]
            return items
