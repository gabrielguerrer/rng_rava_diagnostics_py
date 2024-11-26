"""
Copyright (c) 2024 Gabriel Guerrer

Distributed under the MIT license - See LICENSE for details
"""

"""
This file encompasses the code of the Bytes statistical tests performed
by the Quick and Detailed Tests sub-apps.
"""

import numpy as np
import matplotlib.pyplot as plt

from rng_rava.acq import get_ammount_prefix_str

import rng_rava_diag.tests_tools as tools


def nums_quick_test(nums, num_type, num_min, num_max, float_bins=None):
    num_n = len(nums)

    if num_type == int:
        num_min = int(num_min)
        num_max = int(num_max)
        categories = num_max - num_min + 1
        hist_range = (num_min, num_max + 1)
    else:
        categories = float_bins
        hist_range = (num_min, num_max)

    hist, x = np.histogram(nums, categories, range=hist_range)
    x = x[:-1]
    binom_p = 1/categories
    chisq, chisq_p = tools.nums_bias(nums, categories, hist_range)
    lbl = 'chi2={:.1f}  p={:.1f}%'.format(chisq, chisq_p*100)

    # 95% confidence interval
    ci95, interval = tools.binom_interval(.95, binom_n=num_n, binom_p=binom_p)
    n_exceeding = (hist < ci95[0]).sum() + (hist > ci95[1]).sum()
    lbl_bounds = '{:.1f}% CI'.format(interval*100)
    lbl2 = 'exceed={:.1f}%'.format(n_exceeding/categories*100)

    # Plot
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.set_title('Number bias, n_nums={}, dof={}'.format(get_ammount_prefix_str(num_n), categories-1), loc='left')
    ax.set_ylabel('N')
    ax.plot(x, hist, ls='none', marker='o', ms=3, color='black', label=lbl)
    ax.hlines(ci95, num_min, num_max, ls='dashed', lw=1, color='black', label=lbl_bounds)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    ax2 = ax.twinx()
    ax2.plot(np.nan, np.nan, ls='none', marker='o', ms=3, color='black', label=lbl2)
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc='lower left', bbox_to_anchor=(0, 0))

    return fig


def nums_detailed_test(nums_2d, float_n_bins=100):
    # Load data
    n_tests = nums_2d.shape[0]
    n_nums = nums_2d.shape[1]
    nums_type = int if np.issubdtype(nums_2d.dtype, np.integer) else float
    nums_min = nums_2d.min()
    nums_max = nums_2d.max()

    if nums_type == int:
        n_categ = nums_max + 1 - nums_min
        bins_range = (nums_min, nums_max + 1)
    else:
        n_categ = float_n_bins
        bins_range = (nums_min, nums_max)
    n_dof = n_categ - 1

    # Results arrays
    chi2 = np.zeros(n_tests, dtype=np.float64)

    # Loop and calculate
    for i in range(n_tests):
        chi2[i], _ = tools.nums_bias(nums_2d[i], n_categ, bins_range)

    chi2_dict = tools.fit_chi2(chi2, dof=n_dof)
    chi2_fit_chi2p = chi2_dict['chi2p']
    chi2_fit_dof, chi2_fit_c = chi2_dict['fit_pars']
    chi2_fit_dof_err, chi2_fit_c_err = chi2_dict['fit_pars_stderr']
    chi2_hist, chi2_hist_fit, chi2_err_fit, chi2_bins, chi2_bins_mid = chi2_dict['hist_values']

    lbl_chi2_a = 'dof={:.2f} +- {:.2f}; c={:.0f} +- {:.0f}; p={:.2f}%' \
                 .format(chi2_fit_dof, 1.96*chi2_fit_dof_err, chi2_fit_c, 1.96 * chi2_fit_c_err, chi2_fit_chi2p)

    # Plot
    fig, (ax1) = plt.subplots(1, 1, tight_layout=True, figsize=(5, 4))
    ax1.set_title('Number Bias Distribution, n_tests={}, n_nums/test={}, n_categ={}'
                  .format(get_ammount_prefix_str(n_tests), get_ammount_prefix_str(n_nums), n_categ), loc='left')

    ax1.set_ylabel('N')
    ax1.stairs(chi2_hist, chi2_bins, color='black', lw=1.5, label=lbl_chi2_a)
    ax1.plot(chi2_bins_mid, chi2_hist_fit, color='black', lw=1, ls='--')
    ax1.errorbar(chi2_bins_mid, chi2_hist_fit, 1.96 * chi2_err_fit, color='black', lw=1, ls='none', capsize=2, marker='o', ms=1)
    ymin_1, ymax_1 = ax1.get_ylim()
    ax1.set_ylim(ymin_1, 1.1 * ymax_1)
    ax1.legend(loc='upper right', bbox_to_anchor=(1., 1.))

    return fig