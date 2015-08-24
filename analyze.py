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

def quantilize(xs):
    sx = sorted(xs)
    return [sx.index(x) for x in xs]

def main():
    splits = []
    num_queries = collections.Counter()
    with open('data/results.json') as fin:
        for line in fin:
            result = json.loads(line)

            if 'num_queries' in result:
                num_queries[result['num_queries']] += 1
            splits += result.get('split', [])
    #print(num_queries)
    #exit()

    splits = [Struct(**split) for split in splits]

    #xs = [s.area_fraction for s in splits]
    #ys = [s.pop_fraction for s in splits]
    xs = []
    ys = []
    colors = []
    for s in splits:
        xs.append(s.area_fraction)
        ys.append(s.pop_fraction)
        #a1 = s.area_fraction * s.area
        #a2 = s.area - a1
        #variance_drop = s.variance - \
        #    (s.area_fraction * s.variance1 + (1 - s.area_fraction) * s.variance2)
        #variance_drop = log(s.variance / (s.variance1 + s.variance2 + 1))
        colors.append(s.area_fraction)

    #colors = quantilize(colors)

    pylab.scatter(
        xs, ys, c=colors,
        s=0.5,
        linewidths=(0,),
        cmap='cool')


    pylab.savefig('splits.png', dpi=140)

    #print(splits)


if __name__ == '__main__':
    main()
