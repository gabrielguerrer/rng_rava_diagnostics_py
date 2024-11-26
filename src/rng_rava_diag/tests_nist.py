"""
Copyright (c) 2024 Gabriel Guerrer

Distributed under the MIT license - See LICENSE for details
"""

"""
This file encompasses the code of the NIST's results visualization performed
by the Detailed Tests sub-app.
"""

import numpy as np
import matplotlib.pyplot as plt


def get_txt_par(txt_line, str_i, str_f, par_type=int):
    idx_start = txt_line.index(str_i) + len(str_i)
    idx_end = txt_line.index(str_f)
    par = txt_line[idx_start: idx_end]

    return par_type(par)


def nist_eval(filename_report):
    # Read file
    with open(filename_report, 'r') as f:
        nist_data = f.readlines()

    # Extract ranges
    range_tests = get_txt_par(nist_data[199], 'is approximately = ',  'for a')
    n_stream = get_txt_par(nist_data[200], 'sample size = ',  'binary sequences')

    range_randome = get_txt_par(nist_data[203], 'is approximately = ',  'for a')
    n_randome = get_txt_par(nist_data[203], 'sample size = ',  'binary sequences')

    # Extract tests results
    tests_data = nist_data[7:195]
    n_tests = len(tests_data)

    ps = np.zeros(n_tests, dtype=np.float32)
    props = np.zeros(n_tests, dtype=np.float32)
    tests = []
    for i in range(n_tests):
        line_data = tests_data[i].split(' ')

        while '' in line_data:
            line_data.remove('')

        test = line_data[12].replace('\n', '')
        tests.append(test)

        ps[i] = float(line_data[10])
        props[i] = float(line_data[11].split('/')[0])

        if test in ['RandomExcursions', 'RandomExcursionsVariant']:
            props[i] /= n_randome
        else:
            props[i] /= n_stream

    return ps, props, tests, n_stream, n_randome, range_tests, range_randome


def nist_prop_range(alpha, stream_n):
    p = 1 - alpha
    if stream_n:
        res = 3*np.sqrt(alpha*(1-alpha)/stream_n)
    else:
        res = 0

    return p, res


def nist_test(file_report_a, file_report_b):
    # Parameters
    ALPHA = 0.01

    STS_TESTS = [
    'Frequency',
    'BlockFrequency',
    'Runs',
    'LongestRun',
    'Rank',
    'FFT',
    'Universal',
    'LinearComplexity',
    'Serial',
    'ApproximateEntropy',
    'CumulativeSums',
    'OverlappingTemplate',
    'NonOverlappingTemplate',
    'RandomExcursions',
    'RandomExcursionsVariant'
    ]

    STS_TESTS_SHOW = [
    'Frequency',
    'Block Frequency',
    'Runs',
    'Longest Run',
    'Rank',
    'FFT',
    'Universal',
    'Linear Complexity',
    'Serial (2)',
    'Approximate Entropy',
    'Cumulative Sums (2)',
    'Overlap. Template',
    'Non Overlap. Temp. (148)',
    'Random Exc. (8)',
    'Random Exc. Variant (18)'
    ]

    # RNG A Data
    ps_a, props_a, tests_a, n_stream_a, n_randome_a, range_tests_a, range_randome_a = nist_eval(file_report_a)

    # RNG B Data
    if file_report_b:
        ps_b, props_b, tests_b, n_stream_b, n_randome_b, range_tests_b, range_randome_b = nist_eval(file_report_b)

    # Plot ranges
    mean, rnge = nist_prop_range(ALPHA, n_stream_a)
    mean_exc, range_exc = nist_prop_range(ALPHA, n_randome_a)

    # Plot x
    x_a = np.array([STS_TESTS.index(test) for test in tests_a])
    x_b = x_a + 0.3

    ## PLOT
    MARK_S = 5
    MARK_W = 1
    MARK_F = 'none'

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7.5), tight_layout=True)

    ax1.set_title('NIST Tests Results, SP 800-22 rev. 1a', loc='left')
    ax1.set_ylabel('Proportion')
    ax1.hlines([mean-rnge, min(1., mean+rnge)], 0, 12.5, ls='dotted', color='black', lw=1)
    ax1.hlines([mean_exc-range_exc, min(1., mean_exc+range_exc)], 12.5, 14.5, ls='dotted', color='black', lw=1)
    ax1.plot(x_a, props_a, ls='none', marker='s', markersize=MARK_S, markeredgewidth=MARK_W, color='black', fillstyle=MARK_F, label='A')
    if file_report_b:
        ax1.plot(x_b, props_b, ls='none', marker='8', markersize=MARK_S+1, markeredgewidth=MARK_W, color='black', fillstyle=MARK_F, label='B')
    ax1.legend(loc='lower left', bbox_to_anchor=(0., -0.03), framealpha=0.8)

    # PLOT UNIFORMITY
    ax2.set_xticks(np.arange(15))
    ax2.set_xticklabels(STS_TESTS_SHOW, rotation=90, ha='center')

    ax2.set_ylabel('Uniformity')
    ax2.plot(x_a, ps_a, ls='none', marker='s', markersize=MARK_S, markeredgewidth=MARK_W, color='black', fillstyle=MARK_F)
    if file_report_b:
        ax2.plot(x_b, ps_b, ls='none', marker='8', markersize=MARK_S+1, markeredgewidth=MARK_W, color='black', fillstyle=MARK_F)
    ax2.hlines([.0001, 1.], 0, 15, ls='dotted', color='black', lw=1)

    return fig