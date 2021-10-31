import os
import sys

import numpy as np


def plus_minus(array):
    '''
    Calculates the maximal deviance from the mean.
    '''
    mean = np.mean(array)
    min = np.amin(array)
    max = np.amax(array)
    return np.amax([mean - min, max - mean])

def read_ema(name, count, metric):
    '''
    Opens the numpy arrays with interleaved raw and ema results and
    ignores the raw results to calculate stats.
    Returns:
        The mean, maximal positive and negative deviance from the mean
    '''
    dir = sys.path.append(os.path.realpath('..'))
    cur = []
    mean = []
    plus = []
    minus = []
    for i in range(1, count + 1):
        run = name + str(i)
        cur_both = np.load(os.path.join('tables', 'evaluation', run + metric + '.npy'))
        for j in range(1, len(cur_both), 2):
            cur.append(cur_both[j])
        mean.append(np.mean(cur))
        plus.append(np.amax(cur - np.mean(cur)))
        minus.append(np.mean(cur) - np.amin(cur))
    return mean, plus, minus

def graph_form(name, count, metric, precision, step):
    '''
    Prints the results to the console for copying into tikzpicture graphs.
    '''
    mean, plus, minus = read_ema(name, count, metric)
    print(name + metric)
    for i in range(len(mean)):
        print('(' + str(round((i+1)*step, precision)) + ',' + str(round(mean[i], precision)) + ') +- (' + str(round(plus[i], precision)) + ',' + str(round(minus[i], precision)) + ')')
    print('')

def table_form(name, count, metric, precision):
    '''
    Prints the results to the console for copying into LaTex tables.
    '''
    mean, plus, minus = read_ema(name, count, metric)
    print(name + metric)
    for i in range(len(mean)):
        print('$' + str(round(mean[i], precision)) + '_{\pm ' + str(round(np.amax([plus[i], minus[i]]), precision)) + '}$ & ', end='')
    print('')

for name in {'fixmatch_cifar10hv', 'fixmatch_planktonv'}:
    for metric in {'cross_np', 'center_np', 'acc_np', 'snd_acc_np'}:
        if metric == 'center_np':
            table_form(name, 4, metric, 3)
        else:
            table_form(name, 4, metric, 2)
