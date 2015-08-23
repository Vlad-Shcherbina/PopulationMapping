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

from math import *


def run_solution(command, seed):
    try:
        start = default_timer()
        p = subprocess.Popen(
            'java -jar tester.jar -exec "{}" '
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

            m = re.match(r'## (\w+)(\[\])? = (.*)$', line)
            if m is not None:
                key = m.group(1)
                value = ast.literal_eval(m.group(3))
                if m.group(2):
                    result.setdefault(key, [])
                    result[key].append(value)
                else:
                    assert key not in result
                    result[key] = value

        assert 'score' in result
        return result

    except Exception as e:
        raise Exception('seed={}, out={}, err={}'.format(seed, out, err)) from e


def worker(task):
    return run_solution(*task)


def main():
    subprocess.check_call(
        #'g++ --std=c++11 -Wall -Wno-sign-compare -O2 main.cc -o main',
        'g++ --std=c++0x -W -Wall -Wno-sign-compare '
        '-O2 -s -pipe -mmmx -msse -msse2 -msse3 main.cpp -o main',
        shell=True)
    command = './main'

    tasks = [(command, seed) for seed in range(1, 1000)]

    map = multiprocessing.Pool(10).imap

    average_log_score = 0
    with open('data/results.json', 'w') as fout:
        for result in map(worker, tasks):
            print(result['seed'], result['score'])
            average_log_score += log(result['score'])

            result = json.dumps(result)
            assert '\n' not in result
            fout.write(result + '\n')

    average_log_score /= len(tasks)
    print(average_log_score)


if __name__ == '__main__':
    main()
