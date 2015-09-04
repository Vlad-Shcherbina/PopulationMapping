from math import *
import random
import json
import multiprocessing
import itertools
import sys

import run_many


class Candidate(object):
    def __init__(self, bucket):
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0

        self.mu = -20 + 0.001 * random.random()
        self.sigma = 100.0 + 0.001 * random.random()

        self.bucket = bucket

        self.reserved_seeds = 0


    def accept_result(self, result):
        a = result['adjusted_score']
        self.n += 1
        self.s += a
        self.s2 += a*a

        if self.n >= 10:
            self.mu = self.s / self.n
            sigma = self.s2 / self.n - self.mu**2
            sigma *= (self.n - 1) / self.n
            self.sigma = sqrt(sigma / self.n)


class Bucket(object):
    def __init__(self):
        self.seeds = []
        self.candidates = {}

    def accept_result(self, result):
        assert result['seed'] in self.seeds, result
        self.candidates[result['candidate_label']].accept_result(result)

    def generate_proposal(self):
        best = max(self.candidates, key=lambda k: self.candidates[k].mu + 1e-8 * random.random())
        #new_best = max(self.candidates, key=k: self.candidates[k].mu)
        spec = {}
        for k, c in self.candidates.items():
            spec[k] = random.normalvariate(c.mu, c.sigma)

        spec_best = max(spec, key=spec.get)

        if best == spec_best:
            return -1, None

        weight = spec[spec_best] - spec[best]
        if (self.candidates[best].sigma / (sqrt(self.candidates[best].n) + 1)  >
            self.candidates[spec_best].sigma / (sqrt(self.candidates[spec_best].n) + 1)):
            c = best
        else:
            c = spec_best
        return weight, c
        #return spec[spec_best]

    def show(self):
        for c in sorted(self.candidates, key=lambda c: -self.candidates[c].mu)[:5]:
            cand = self.candidates[c]
            print('{:>4} {:>25}: {:.4} +- {:.4}'.format(
                cand.n, repr(c), cand.mu, cand.sigma))

    def best(self):
        c = max(self.candidates, key=lambda c: self.candidates[c].mu)
        cand = self.candidates[c]
        return c, cand.mu, cand.sigma


class Optimizer(object):
    def __init__(self, seeds_by_bucket, candidate_labels):
        self.buckets = {}
        for bucket_label, seeds in seeds_by_bucket.items():
            self.buckets[bucket_label] = b = Bucket()
            b.seeds = seeds
            for c in candidate_labels:
                b.candidates[c] = Candidate(b)

    def accept_result(self, result):
        self.buckets[result['bucket_label']].accept_result(result)

    def generate_tasks(self, n):
        while True:
            proposals = {k: b.generate_proposal() for k, b in self.buckets.items()}
            best = max(proposals, key=lambda p: proposals[p][0])
            if proposals[best][0] > -1:
                break

        bucket = self.buckets[best]
        candidate_label = proposals[best][1]
        candidate = bucket.candidates[candidate_label]

        seeds = bucket.seeds[
            candidate.reserved_seeds: candidate.reserved_seeds + n]
        candidate.reserved_seeds += n
        assert seeds

        return [(seed, candidate_label) for seed in seeds]

    def advance(self, imap):
        tasks = sum((self.generate_tasks(10) for _ in range(40)), [])
        tasks = [('./main {} {} {}'.format(*candidate_label),
                  seed,
                  dict(candidate_label=candidate_label))
                 for seed, candidate_label in tasks]

        for result in imap(run_many.worker, tasks):
            result['bucket_label'] = (
                result['percentage_bucket'], result['land_density_bucket'])
            self.accept_result(result)

    def show(self):
        total_mu = 0
        total_sigma = 0
        best_sol = []
        for k, b in sorted(self.buckets.items()):
            print('-------- {} --------'.format(repr(k)))
            b.show()

            c, mu, sigma = b.best()
            best_sol.append([list(k), list(c)])

            total_mu += mu
            total_sigma += sigma**2

        total_mu /= len(self.buckets)
        total_sigma /= len(self.buckets)
        total_sigma = sqrt(total_sigma)

        print(repr(best_sol).replace('[', '{').replace(']', '}'))
        print('*' * 30)
        print('TOTAL SCORE: {:.4} +- {:.4}'.format(total_mu, total_sigma))
        print('*' * 30)
        sys.stdout.flush()



def main():
    run_many.build()

    with open('data/seeds_by_bucket.json') as fin:
        seeds_by_bucket = json.load(fin)
    seeds_by_bucket = {tuple(k): v for k, v in seeds_by_bucket}

    pool = multiprocessing.Pool(12)

    candidate_labels = [(2, 1, 1), (0.643, 0.689, 1)]
    for _ in range(50):
        candidate_labels.append((
            round(0.001 * random.randrange(600, 2000), 3),
            round(0.001 * random.randrange(600, 2000), 3),
            round(0.001 * random.randrange(500, 2000), 3),
            ))

    opt = Optimizer(seeds_by_bucket, candidate_labels)
    while True:
        print('*' * 50)
        opt.show()
        opt.advance(pool.imap)


if __name__ == '__main__':
    main()
