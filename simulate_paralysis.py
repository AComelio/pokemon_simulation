import random
import math
import time
from itertools import repeat
from scipy.stats import binom, poisson

import multiprocessing as mp

def time_me(func):
    # A simple timing wrapper to add time to run reporting to a function
    def inner1(*args, **kwargs):
        st = time.perf_counter_ns()
        r = func(*args, **kwargs)
        print(f'Function done in {(time.perf_counter_ns()-st)/1e9} seconds real time')
        return r
    return inner1

@time_me
def austin_original(trials):
    # Austin's original solution, included for comparison in speed to my solutions

    items = [1,2,3,4]
    numbers = [0,0,0,0]
    rolls = 0
    maxOnes = 0

    while numbers[0] < 177 and rolls < trials:
        numbers = [0,0,0,0]
        for i in repeat(None, 231):
            roll = random.choice(items)
            numbers[roll-1] = numbers[roll-1] + 1
        rolls = rolls + 1
        if numbers[0] > maxOnes:
            maxOnes = numbers[0]

    return maxOnes

@time_me
def neatened_austin_style(trials):
    # A solution in the spirit of Austin's original, re-written in a more efficient style


    max_sucesses = 0
    for _ in range(trials):
        # my favourite python trick, True/False get treated as 1/0 in arithmetic, so sum(True, True, False) is 2
        sucesses = sum(random.random() < 0.25 for _ in range(231))
        if sucesses > max_sucesses:
            max_sucesses = sucesses
    return max_sucesses

@time_me
def binomial_style(trials):
    # A solution sampling the binomial distribution utilising a C implemented function in random
    return max(random.binomialvariate(231, 0.25) for _ in range(trials))

@time_me
def sampled_binomial(trials):
    # Best solution so far, pre-computes the probability of getting at least x sucesses in 231 trials for x = 0 to 231
    # These probabilities are used as weights for random.choices to select from the list of 0 to 231 a trials number of times

    vals = list(range(232))
    weights = [1 - binom.cdf(k, 231, 0.25) for k in vals]
    return max(random.choices(vals, weights, k=trials))

def unpack_for_choices(bundle):
    # Same as sampled binomial without wrapper function or weights gen for efficient multiprocessing

    vals, weights, trials = bundle
    return max(random.choices(vals, weights, k=trials))

@time_me
def multiprocessed_binomial_style(trials):
    # This solution takes the previous solution and distributes the work across multiple cpu cores - valid because each trial is independent
    # Due to the overhead of setting up new python processes, this solution is slower for small numbers of trials
    # For large numbers of trials, provides a factor approximately proportional to cpu cores, i.e. 8 cores = 8 times faster

    # This defines how many sub-calls will be made, should be an exact divisor of trials or there will be slight undersampling
    blocks = 1000

    vals = list(range(232))
    weights = [1 - binom.cdf(k, 231, 0.25) for k in range(232)]
    sub_trials = trials//blocks

    # packages up the values required for random.choices a blocks number of times
    gen = ((vals, weights, sub_trials) for _ in range(blocks))

    # Defines a pool of python child processes equal to cpu count, and submits data to them for processing
    with mp.Pool() as p:
        m = max(p.imap_unordered(unpack_for_choices, gen))
    return m

def main():

    # Lets me test different trial values, observed complexity is O(n), i.e. linear scaling, with the caveat multiprocessing has a minimum time for low trial counts
    trials = [
              100_000,
            1_000_000,
           10_000_000,
          100_000_000,
        1_000_000_000
    ]

    print(f'Commencing Austin original function with {trials[0]:_} trials')
    print(austin_original(trials[0]), 'max sucesses')

    print(f'Commencing Austin style function with {trials[0]:_} trials')
    print(neatened_austin_style(trials[0]), 'max sucesses')

    for trial in trials[:3]:
        print(f'Commencing binomial style function with {trial:_} trials')
        print(binomial_style(trial), 'max sucesses')

    for trial in trials[2:]:
        print(f'Commencing sampled binomial function with {trial:_} trials')
        print(sampled_binomial(trial), 'max sucesses')

    for trial in trials[2:]:
        print(f'Commencing multiprocessed function with {trial:_} trials')
        print(multiprocessed_binomial_style(trial), 'max sucesses')

if __name__ == '__main__':
    # Boilerplate required for multiprocessing
    main()
