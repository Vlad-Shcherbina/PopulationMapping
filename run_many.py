import ast
import sys
import multiprocessing
import os
import re
import subprocess
from timeit import default_timer
import multiprocessing
import pprint
import json
import random

from math import *


def run_solution(command, seed):
    try:
        start = default_timer()
        p = subprocess.Popen(
            'java -classpath tester/patched_tester.jar PopulationMappingVis '
            '-exec "{}" '
            '-novis -seed {}'.format(command, seed),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out, err = p.communicate()
        out = out.decode()
        err = err.decode()

        p.wait()
        assert p.returncode == 0

        result = dict(
            seed=str(seed),
            time=default_timer() - start)

        for line in out.splitlines() + err.splitlines():
            m = re.match(r'Score = (.+)$', line)
            if m is not None:
                result['score'] = float(m.group(1))

            m = re.match(r'## ([\w.]+)(\[\])? = (.*)$', line)
            if m is not None:
                key = m.group(1)
                try:
                    value = ast.literal_eval(m.group(3))
                except Exception as e:
                    print(e)
                    continue
                if m.group(2):
                    result.setdefault(key, [])
                    result[key].append(value)
                else:
                    assert key not in result
                    result[key] = value

        #assert 'score' in result
        assert result['score'] > 0

        if 'land_area' in result and 'max_percentage' in result:
            result['adjusted_score'] = log(result['score'] /
                (result['land_area'] * pow(result['max_percentage'] * 0.01, 0.6)))
        else:
            print('qwasdf')
        return result

    except Exception as e:
        raise Exception('seed={}, out={}, err={}'.format(seed, out, err)) from e


def worker(task):
    return run_solution(*task)


def build():
    subprocess.check_call(
        #'g++ --std=c++11 -Wall -Wno-sign-compare -O2 main.cc -o main',
        'g++ --std=c++0x -W -Wall -Wno-sign-compare -D NDEBUG'
        '-O2 -s -pipe -mmmx -msse -msse2 -msse3 main.cpp -o main',
        shell=True)


def main():
    command = './main {} {}'

    # tasks = [
    #     (command.format(random.uniform(1.8, 2.2), random.uniform(0.9, 1.1)), seed)
    #     for seed in range(1, 20000)]
    tasks = [
        (command.format(0.85, 0.65), seed)
        for seed in range(1, 500)]

    map = multiprocessing.Pool(12).imap

    average_adjusted_score = 0
    with open('data/results.json', 'w') as fout:
        for result in map(worker, tasks):
            print(result['seed'], result['score'])
            average_adjusted_score += result.get('adjusted_score', -100000)

            result = json.dumps(result)
            assert '\n' not in result
            fout.write(result + '\n')

    average_adjusted_score /= len(tasks)
    print(average_adjusted_score * 100)


if __name__ == '__main__':
    main()
