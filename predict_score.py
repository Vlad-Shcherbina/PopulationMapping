import json
import collections
import itertools

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
    results = []
    with open('data/results.json') as fin:
        for line in fin:
            result = json.loads(line)
            #frac = result['land_area'] / (result['W'] * result['H'])
            #if frac > 0.2:
            results.append(result)

    print(len(results), 'data points')

    features = collections.defaultdict(list)
    for result in results:
        for k, v in result.items():
            features[k].append(v)

    features = Struct(**{
        k: linear_regression.RawFeature(k, numpy.array(v, dtype=numpy.double))
        for k, v in features.items()})

    #print(collections.Counter(features.land_density_bucket.xs))
    #print(collections.Counter(features.percentage_bucket.xs))

    goal = 100 * numpy.log(features.score.xs / features.land_area.xs)

    land_fraction = linear_regression.RawFeature(
        'land_fraction',
        (features.land_area.xs / (features.variance_x.xs + features.variance_y.xs)))


    #land_fraction = RawFeature('land_fraction', features.land_area / (features.w * features.h))

    #print(sorted(land_fraction.xs)[::len(land_fraction.xs) // 5])
    q = sorted(land_fraction.xs)
    print(q[-1])
    print([q[i * (len(q) - 1) // 5] for i in range(5 + 1)])


    log_area = linear_regression.RawFeature(
        'log_area',
        numpy.log(features.land_area.xs))

    #fs = land_fraction
    #print(land_fraction)
    ones = linear_regression.ConstantOneFeature(len(goal))

    elongation = linear_regression.RawFeature(
        'elongation',
        (numpy.log(features.W.xs / features.H.xs) ** 2))

    density = linear_regression.RawFeature(
        'density',
        (features.total_population.xs / features.land_area.xs))

    adjusted_percentage = linear_regression.RawFeature(
        'adjusted_percentage',
        (0.01 * features.max_percentage.xs) ** 0.1)

    #land_fraction.xs **= 2
    #features.variance_x.xs **= 0.5
    #features.variance_y.xs **= 0.5

    fs = [land_fraction, adjusted_percentage]
    fs = [ones] + [linear_regression.standardize_feature(f) for f in fs]

    fs = linear_regression.polynomial_features(fs, 3)

    lr = linear_regression.LinearRegression.create(fs, goal)

    lr.solve()

    print(lr.solution)
    for k, f in sorted(zip(lr.solution, lr.features), key=lambda q: abs(q[0])):
        print('+', k, '*', f.get_expression())

    print('penalty', sqrt(lr.penalty / len(goal)))
    print(len(goal))


    #print(lr.solution)
    #for k, f in sorted(zip(lr.solution, lr.features), key=lambda q: abs(q[0])):
    #    print('{:15.5} {}'.format(k, f.get_expression()))


    #print(goal)
    #print(numpy.std(goal[:1000]))


if __name__ == '__main__':
    main()
