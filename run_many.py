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
import collections

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

        land_fraction = result['land_area'] / (result['variance_x'] + result['variance_y'])
        adjusted_percentage = (0.01 * result['max_percentage']) ** 0.1
        result['adjusted_score'] = 100 * log(result['score'] / result['land_area']) - (
            + -0.0211455093991 * (0.9438674*land_fraction+-1.531457)*(0.9438674*land_fraction+-1.531457)*(0.9438674*land_fraction+-1.531457)
            + -0.232864716103 * (13.24757*adjusted_percentage+-11.55022)*(13.24757*adjusted_percentage+-11.55022)*(13.24757*adjusted_percentage+-11.55022)
            + -0.330680069254 * (0.9438674*land_fraction+-1.531457)*(13.24757*adjusted_percentage+-11.55022)*(13.24757*adjusted_percentage+-11.55022)
            + 0.819358049411 * 1*(0.9438674*land_fraction+-1.531457)*(0.9438674*land_fraction+-1.531457)
            + -1.05737826196 * (0.9438674*land_fraction+-1.531457)*(0.9438674*land_fraction+-1.531457)*(13.24757*adjusted_percentage+-11.55022)
            + 4.52962891409 * 1*(0.9438674*land_fraction+-1.531457)*(13.24757*adjusted_percentage+-11.55022)
            + -4.88050957678 * 1*1*(0.9438674*land_fraction+-1.531457)
            + -7.67401082496 * 1*(13.24757*adjusted_percentage+-11.55022)*(13.24757*adjusted_percentage+-11.55022)
            + 52.5224028358 * 1*1*(13.24757*adjusted_percentage+-11.55022)
            + -68.4225994083 * 1*1*1
        )


        #if 'land_area' in result and 'max_percentage' in result:
        #    result['adjusted_score'] = log(result['score'] /
        #        (result['land_area'] * pow(result['max_percentage'] * 0.01, 0.6)))
        #else:
        #    print('qwasdf')
        return result

    except Exception as e:
        raise Exception('seed={}, out={}, err={}'.format(seed, out, err)) from e


def worker(task):
    if len(task) == 2:
        command, seed = task
        return run_solution(command, seed)
    else:
        command, seed, extra = task
        result = run_solution(command, seed)
        result.update(extra)
        return result


def build():
    subprocess.check_call(
        #'g++ --std=c++11 -Wall -Wno-sign-compare -O2 main.cc -o main',
        'g++ --std=c++0x -W -Wall -Wno-sign-compare -D NDEBUG1'
        '-O2 -s -pipe -mmmx -msse -msse2 -msse3 main.cpp -o main',
        shell=True)


def main():
    #build()

    #command = './main {} {}'

    # tasks = [
    #     (command.format(random.uniform(1.8, 2.2), random.uniform(0.9, 1.1)), seed)
    #     for seed in range(1, 20000)]
    tasks = [
        ('./main', seed)
        for seed in range(1, 250000)]

    map = multiprocessing.Pool(12).imap

    average_adjusted_score = 0

    seeds_by_bucket = collections.defaultdict(list)

    with open('data/results_.json', 'w') as fout:
        for i, result in enumerate(map(worker, tasks)):
            bucket = (
                result['percentage_bucket'], result['land_density_bucket'])
            seeds_by_bucket[bucket].append(result['seed'])

            print(result['seed'], result['score'])
            average_adjusted_score += result.get('adjusted_score', -100000)

            result = json.dumps(result)
            assert '\n' not in result
            fout.write(result + '\n')

            if i % 100 == 0:
                with open('data/seeds_by_bucket_.json', 'w') as buckets_fout:
                    json.dump(list(seeds_by_bucket.items()), buckets_fout)

    average_adjusted_score /= len(tasks)
    print('score:', average_adjusted_score)

    #seeds_by_bucket = dict(seeds_by_bucket)
    for k, v in seeds_by_bucket.items():
        print(k, v)
    print(len(seeds_by_bucket))
    with open('data/seeds_by_bucket_.json', 'w') as fout:
        json.dump(list(seeds_by_bucket.items()), fout)


if __name__ == '__main__':
    main()
