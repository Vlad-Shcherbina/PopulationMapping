import json
import collections

from matplotlib import pylab
import random

from math import *

# from http://norvig.com/python-iaq.html
class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)

    def __repr__(self):
        args = ['%s=%s' % (k, repr(v)) for (k,v) in vars(self).items()]
        return 'Struct(%s)' % ', '.join(args)

def main():
    splits = []
    with open('data/results.json') as fin:
        for line in fin:
            result = json.loads(line)
            splits += result.get('split', [])

    splits = [Struct(**split) for split in splits]

    buckets = collections.defaultdict(list)
    buckets[0] = splits
    # for split in splits:
    #     r = abs(log(split.w / split.h))
    #     buckets[int(r * 3)].append(split)

    for i, (_, bucket) in enumerate(sorted(buckets.items())):
        xs = [s.area_fraction for s in bucket]
        ys = [s.pop_fraction for s in bucket]
        pylab.scatter(
            xs, ys, s=0.01,
            color=(1.0 * i / len(buckets), 0, 0))


    pylab.savefig('splits.png', dpi=140)

    #print(splits)


if __name__ == '__main__':
    main()
