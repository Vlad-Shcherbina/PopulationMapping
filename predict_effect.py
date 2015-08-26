import json
import collections
import itertools

from matplotlib import pylab
import random

from math import *
import numpy
from numpy import linalg

import linear_regression


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

    print(len(splits))
    ss = splits[:]
    splits = []
    for split in ss:
        area = split['child1.area'] + split['child2.area']
        assert area >= 1, split
        assert 0 <= split['actual_effect'] <= area, split

        slope = split['parent.pop'] / area
        if slope >= split['slope']:
            splits.append(split)
    print(len(splits))

    #splits = [Struct(**split) for split in splits]

    features = collections.defaultdict(list)
    for split in splits:
        for k, v in split.items():
            features[k].append(v)

    effect = features['actual_effect']
    del features['actual_effect']

    del features['hz']

    del features['expected_effect']

    features = {
        k: linear_regression.RawFeature(k, numpy.array(v, dtype=numpy.double))
        for k, v in features.items()}
    #features = Struct(**features)
    #print(features)

    #features['inv_slope'] = linear_regression.RawFeature(
    #    '1/slope', 1.0/features['slope'].xs)

    area = features['child1.area'].xs + features['child2.area'].xs
    effect /= area

    #features['parent.pop'].xs /= area
    # for k in [
    #     'child1.stdev_x',
    #     'child1.stdev_y',
    #     'child2.stdev_x',
    #     'child2.stdev_y',
    #     'parent.stdev_x',
    #     'parent.stdev_x',
    #     ]:
    #     features[k].xs /= numpy.sqrt(area)

    ones = linear_regression.ConstantOneFeature(len(effect))

    features = [linear_regression.standardize_feature(f)
        for _, f in sorted(features.items())]
    features.append(ones)
    features = linear_regression.polynomial_features(features, 3)
    #print(features)
    print(len(features), 'features')

    lr = linear_regression.LinearRegression.create(
        features,
        effect,
        weights=area)

    lr.solve()

    #print(lr.solution)
    #for k, f in sorted(zip(lr.solution, lr.features), key=lambda q: abs(q[0])):
    #    print(k, f.get_expression())

    print('penalty', sqrt(lr.penalty / len(effect)))

    xs = []
    ys = []
    colors = []
    for x, y, a in zip(lr.predicted_results(), effect, area):
        xs.append(x)
        ys.append(y)
        colors.append(a)

    def quantilize(xs):
        sx = sorted(xs)
        return [sx.index(x) for x in xs]
    #colors = quantilize(colors)


    pylab.plot([0, 0.3], [0, 0.3], color='red')
    pylab.scatter(
        xs, ys, c=colors,
        s=0.5,
        linewidths=(0,),
        cmap='cool')
    pylab.axis([-0.1, 0.4, 0, 0.4])
    pylab.savefig('linear_regression.png', dpi=140)
    #print(features)


    model = ' + '.join(
        '{:.7}*{}'.format(a, f.get_expression())
        for a, f in zip(lr.solution, lr.features))
    model = 'return (child1.area + child2.area) * ({});\n'.format(model)
    with open('model.txt', 'w') as fout:
        fout.write(model)


if __name__ == '__main__':
    main()
