import json
import collections
import statistics
import random
from math import *

from matplotlib import pylab


# from http://norvig.com/python-iaq.html
class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)

    def __repr__(self):
        args = ['%s=%s' % (k, repr(v)) for (k,v) in vars(self).items()]
        return 'Struct(%s)' % ', '.join(args)

def quantilize(xs):
    sx = sorted(xs)
    return [sx.index(x) for x in xs]

def main():
    splits = []
    with open('data/results.json') as fin:
        for line in fin:
            result = json.loads(line)
            splits += result.get('split', [])

    splits = [Struct(**split) for split in splits]

    NUM_BUCKETS = 22

    by_bucket = collections.defaultdict(list)
    for s in splits:
        bucket = int(s.area_fraction * NUM_BUCKETS + 0.5) / NUM_BUCKETS
        by_bucket[bucket].append(s.pop_fraction)

    by_bucket = dict(by_bucket)

    total_variance = 0
    for bucket, xs in sorted(by_bucket.items()):
        if len(xs) < 2:
            continue
        print('{}:   {:.4f} +- {:.4f} ({})'.format(
            bucket,
            statistics.mean(xs), statistics.stdev(xs),
            len(xs)))

        total_variance += statistics.variance(xs) * len(xs) / len(splits)

    print('total stddev', sqrt(total_variance))

    #print(by_bucket[0.3])

    xs = by_bucket[0.0]

    n, bins, patches = pylab.hist(xs, 20, normed=1, histtype='stepfilled')
    pylab.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

    #mu = statistics.mean(xs)
    mu = statistics.mode(int(x * 20 + 0.5) / 20 for x in xs)
    sigma = statistics.pstdev(xs, mu=mu)
    # add a line showing the expected distribution
    y = pylab.normpdf(bins, mu, sigma)
    l = pylab.plot(bins, y, 'k--', linewidth=1.5)

    #by_bucket[x]

    #pylab.scatter(
    #    xs, ys, c=colors,
    #    s=0.1,
    #    linewidths=(0,),
    #    cmap='cool')


    pylab.savefig('pop_fraction.png', dpi=140)


    result = []
    for bucket, xs in sorted(by_bucket.items()):
        N = 36
        xs = sorted(xs)
        #result.append(str([xs[(len(xs) - 1) * i // N] for i in range(N + 1)]))
        result.append(str([
            statistics.mean(xs[len(xs) * i // N : len(xs) * (i + 1) // N])
            for i in range(N)
            ]))

    result = ', '.join(r.replace('[', '{').replace(']', '}') for r in result)
    with open('prediction.h', 'w') as fout:
        fout.write('const vector<vector<double>> prediction = {{ {} }};'.format(result))

    #print(splits)


if __name__ == '__main__':
    main()
