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
    results = []
    with open('data/results.json') as fin:
        for line in fin:
            result = json.loads(line)
            results.append(Struct(**result))

    xs = []
    ys = []
    colors = []
    for result in results:
        if result.adjusted_score < 0 or result.adjusted_score > 0.2:
            continue
        xs.append(result.rank_split_alpha)
        ys.append(result.rank_split_beta)
        colors.append(result.adjusted_score)

    pylab.scatter(xs, ys, c=colors, s=20, marker='.', linewidths=(0,))
    pylab.savefig('results.png', dpi=140)

    #print(splits)


if __name__ == '__main__':
    main()
