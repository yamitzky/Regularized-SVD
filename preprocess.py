from random import random


def preprocess():
    with open('data/ratings.dat') as f:
        with open('data/u1.base', 'w') as w1:
            with open('data/u1.test', 'w') as w2:
                for line in f:
                    line = '\t'.join(line.strip().split('::')) + '\n'
                    if random() > 0.2:
                        w1.writelines(line)
                    else:
                        w2.writelines(line)


if __name__ == '__main__':
    preprocess()
