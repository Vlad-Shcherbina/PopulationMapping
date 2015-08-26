import itertools

import numpy
from numpy import linalg


class Feature(object):
    def get_preamble(self):
        return ''

    def get_expression(self):
        raise NotImplementedError()

    def __repr__(self):
        return '<feature>({}={})'.format(self.get_expression(), self.xs)


class RawFeature(Feature):
    def __init__(self, label, xs):
        self.label = label
        self.xs = numpy.array(xs, dtype=numpy.double)

    def get_expression(self):
        return self.label


class ConstantOneFeature(Feature):
    def __init__(self, size):
        self.xs = numpy.ones(size)

    def get_expression(self):
        return '1'


_id = 0

class LinearlyTransformedFeature(Feature):
    def __init__(self, feature, a, b):
        global _id
        self.a = a
        self.b = b
        self.feature = feature
        self.xs = a * feature.xs + b
        self.var_name = '_x{}'.format(_id)
        _id += 1

    def get_preamble(self):
        result = self.feature.get_preamble()
        result += 'double {} = {} * {} + {};\n'.format(
            self.var_name, self.a, self.feature.get_expression(), self.b)
        return result

    def get_expression(self):
        return '({:.7}*{}+{:.7})'.format(self.a, self.feature.get_expression(), self.b)
        #return self.var_name


def standardize_feature(feature):
    mean = numpy.mean(feature.xs)
    stdev = numpy.std(feature.xs)
    if stdev > 1e-8:
        a = 1.0 / stdev
    else:
        a = 1.0
    return LinearlyTransformedFeature(feature, a, -mean*a)


def build_term(features):
    xs = 1
    for f in features:
        xs = xs * f.xs
    # TODO
    return RawFeature('*'.join(f.get_expression() for f in features), xs)


def polynomial_features(features, degree):
    result = []
    for t in itertools.combinations_with_replacement(features, degree):
        result.append(build_term(t))
    return result


class LinearRegression(object):
    @staticmethod
    def create(features, values, weights=None):
        lr = LinearRegression()
        lr.features = list(features)

        n = len(lr.features)

        aT = numpy.array([f.xs for f in lr.features])
        y = numpy.array(values, dtype=numpy.double)

        if weights is not None:
            aT *= weights
            y *= weights

        lr.aTa = numpy.zeros((n, n))
        lr.aTy = numpy.zeros(n)

        for i in range(n):
            lr.aTy[i] = aT[i].dot(y)
            for j in range(i, n):
                lr.aTa[i, j] = lr.aTa[j, i] = aT[i].dot(aT[j])

        lr.yTy = y.dot(y)
        return lr

    def solve(self, regularization=0.0):
        # http://stackoverflow.com/questions/27476933/numpy-linear-regression-with-regularization
        sol, residuals, rank, s = linalg.lstsq(
            self.aTa + regularization * numpy.identity(len(self.aTy)),
            self.aTy)
        assert len(sol) == len(self.features)

        self.penalty = \
            sol.dot(self.aTa).dot(sol) - 2 * sol.dot(self.aTy) + self.yTy

        self.solution = sol
        return sol

    def predicted_results(self):
        result = 0
        for a, f in zip(self.solution, self.features):
            result += a * f.xs
        return result
