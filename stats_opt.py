from math import *
import random
import itertools
import multiprocessing
import sys


#import time

import run_many


N = 200


def cumulative_mu_sigma(xs):
    n = 0
    s = 0.0
    s2 = 0.0
    for x in xs:
        s += x
        s2 += x*x
        n += 1

        if n > 0 and n % N == 0:
            mean = s / n
            sigma = s2 / n - mean * mean
            sigma *= (n - 1) / n
            sigma = sqrt(sigma)

            yield mean, sigma / sqrt(n)



class C(object): pass

def thompson_tournament(candidates):
    #candidates = [for label, xs in candidates.items()]
    state = {}
    for label, xs in candidates.items():
        state[label] = c = C()
        c.stats = iter(cumulative_mu_sigma(xs))
        #c.mu, c.sigma = next(c.stats)
        c.mu = 0.0
        c.sigma = 100.0
        c.n = 0

    while True:
        for label in sorted(state,
                            key=lambda label: state[label].mu, reverse=True):
            c = state[label]
            print('{:5} {:>30}: {:.4} +- {:.6}'.format(
                c.n, repr(label), c.mu, c.sigma))


        def rnd(label):
            c = state[label]
            return random.normalvariate(c.mu, c.sigma)

        def advance(label):
            c = state[label]
            c.mu, c.sigma = next(c.stats)
            c.n += 1

        to_advance = max(state, key=rnd)
        print('advancing', to_advance)
        sys.stdout.flush()
        advance(max(state, key=rnd))


        #time.sleep(0.1)



def main():
    run_many.build()

    command = './main {} {}'

    pool = multiprocessing.Pool(12)


    def make_candidate(*args):
        def gen():
            while True:
                tasks = [
                    (command.format(*args), random.randrange(1, 1000000000))
                    for _ in range(N)]
                results = list(pool.imap(run_many.worker, tasks))
                for result in results:
                    yield result['adjusted_score'] * 100

        return args, gen()

    #candidates = dict(map(make_candidate, [(0.85, 0.65), (2, 1)]))
    candidates = []
    for _ in range(50):
        candidates.append(make_candidate(
            random.uniform(0.5, 1),
            random.uniform(0.4, 0.9),
            ))
        candidates.append(make_candidate(
            random.uniform(1.5, 2.5),
            random.uniform(0.7, 1.8),
            ))


    thompson_tournament(dict(candidates))


if __name__ == '__main__':
    main()
