"""
Copyright (c) 2024 Gabriel Guerrer

Distributed under the MIT license - See LICENSE for details
"""

"""
This file encompasses the code of the Pulse Counts statistical tests performed
by the Quick and Detailed Tests sub-apps.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

from rng_rava.acq import get_ammount_prefix_str

import rng_rava_diag.tests_tools as tools


def fmt_plot_thbias(x, y, x_flat, y_flat, z_flat):
    # Get closest point with known data
    dist = np.linalg.norm(np.vstack([x_flat - x, y_flat - y]), axis=0)
    idx = np.argmin(dist)
    z = z_flat[idx]
    return 'x={x:.2f}  y={y:.2f}  z={z:.2e}'.format(x=x, y=y, z=z)


def pcs_thbias():
    n_points = 100
    mu_range = np.linspace(0, 32, n_points)
    std_range = np.linspace(.5, 3,  n_points)
    mu, std = np.meshgrid(mu_range, std_range, indexing='ij')

    # Calculate Theoretical Bias
    bias = np.zeros(shape=(len(mu_range), len(std_range)), dtype=np.float64)
    for i, mu_ in enumerate(mu_range):
        for j, std_ in enumerate(std_range):
            bias[i, j] = tools.theoretical_bit_bias(mu_, std_)

    # Replace 0 with the min value -> caused by the float64 precision
    bias_new = bias.copy()
    bias_min = np.min(bias_new[np.where(bias_new>0)])
    bias_new[np.where(bias_new==0)] = bias_min

    # Plot
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5), tight_layout=True)

    ax1.set_title('Theoretical Bit Bias / Normal pars', loc='left')
    ax1.set_xlabel('mu')
    ax1.set_ylabel('std')
    cs = ax1.contourf(mu, std, bias_new, locator=ticker.LogLocator(), cmap=cm.gray)
    cbar = fig.colorbar(cs, label='Bias')
    cbar.ax.set_yscale('log')

    x_flat = mu.flatten()
    y_flat = std.flatten()
    z_flat = bias.flatten()
    ax1.format_coord = lambda x, y: fmt_plot_thbias(x, y, x_flat, y_flat, z_flat)

    return fig


def pcs_quick_test(pcs_1d_a, pcs_1d_b, rng_setup_str=''):
    n_pcs = len(pcs_1d_a)

    # PCs A: Fit histogram
    res_dict_a = tools.fit_normal(pcs_1d_a)
    chi2p_a = res_dict_a['chi2p']
    norm_mu_a, norm_std_a, norm_c_a = res_dict_a['fit_pars']
    pcs_hist_a, pcs_hist_fit_a, pcs_err_fit_a, pcs_bins_a, pcs_bins_mid_a = res_dict_a['hist_values']
    lbl_a = 'RNG A: mu={:.2f}, std={:.2f}; chi2_p={:.1f}%'.format(norm_mu_a, norm_std_a, chi2p_a)

    # PCs B: Fit histogram
    res_dict_b = tools.fit_normal(pcs_1d_b)
    chi2p_b = res_dict_b['chi2p']
    norm_mu_b, norm_std_b, norm_c_b = res_dict_b['fit_pars']
    pcs_hist_b, pcs_hist_fit_b, pcs_err_fit_b, pcs_bins_b, pcs_bins_mid_b = res_dict_b['hist_values']
    lbl_b = 'RNG B: mu={:.2f}, std={:.2f}; chi2_p={:.1f}%'.format(norm_mu_b, norm_std_b, chi2p_b)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(10, 4))
    ax1.set_title('Pulse Count Distribution, n_pcs={}; {}'.format(get_ammount_prefix_str(n_pcs), rng_setup_str), loc='left')

    ax1.set_ylabel('N')
    ax1.stairs(pcs_hist_a, pcs_bins_a, color='black', lw=1.5, label=lbl_a)
    ax1.plot(pcs_bins_mid_a, pcs_hist_fit_a, color='black', lw=1, ls='--')
    ax1.errorbar(pcs_bins_mid_a, pcs_hist_fit_a, 1.96 * pcs_err_fit_a, color='black', lw=1, ls='none', capsize=2, marker='o', ms=1)
    ymin_1, ymax_1 = ax1.get_ylim()
    ax1.set_ylim(ymin_1, 1.1 * ymax_1)
    ax1.legend(loc='upper right', bbox_to_anchor=(1., 1.))

    ax2.set_ylabel('N')
    ax2.stairs(pcs_hist_b, pcs_bins_b, color='gray', lw=1.5, label=lbl_b)
    ax2.plot(pcs_bins_mid_b, pcs_hist_fit_b, color='gray', lw=1, ls='--')
    ax2.errorbar(pcs_bins_mid_b, pcs_hist_fit_b, 1.96 * pcs_err_fit_b, color='gray', lw=1, ls='none', capsize=2, marker='o', ms=1)
    ymin_2, ymax_2 = ax2.get_ylim()
    ax2.set_ylim(ymin_2, 1.1 * ymax_2)
    ax2.legend(loc='upper right', bbox_to_anchor=(1., 1.))

    return fig


def pcs_fit_normal(pcs_2d):
    n_repeat = pcs_2d.shape[0]
    chi2p = np.zeros(n_repeat)
    norm_mu = np.zeros(n_repeat)
    norm_std = np.zeros(n_repeat)
    norm_c = np.zeros(n_repeat)

    for i in range(n_repeat):
        pcs = pcs_2d[i,:]

        # Fit to normal distrib
        res_dict = tools.fit_normal(pcs)
        chi2p[i] = res_dict['chi2p']
        norm_mu[i], norm_std[i], norm_c[i] = res_dict['fit_pars']

    return chi2p, (norm_mu, norm_std, norm_c)


def pcs_detailed_test(si_range, pcs_3d_a, pcs_3d_b):
    # Data
    n_si = pcs_3d_a.shape[0]
    n_tests = pcs_3d_a.shape[1]
    n_pcs = pcs_3d_a.shape[2]

    pool = ProcessPoolExecutor()
    fut_a = []
    fut_b = []

    chi2p_a = np.zeros(shape=(n_si, n_tests))
    norm_mu_a = np.zeros(shape=(n_si, n_tests))
    norm_std_a = np.zeros(shape=(n_si, n_tests))
    norm_c_a = np.zeros(shape=(n_si, n_tests))

    chi2p_b = np.zeros(shape=(n_si, n_tests))
    norm_mu_b = np.zeros(shape=(n_si, n_tests))
    norm_std_b = np.zeros(shape=(n_si, n_tests))
    norm_c_b = np.zeros(shape=(n_si, n_tests))

    # Fit normal distrib
    for i, si in enumerate(si_range):
        fut_a.append( pool.submit(pcs_fit_normal, pcs_3d_a[i]) )
        fut_b.append( pool.submit(pcs_fit_normal, pcs_3d_b[i]) )

    for i, si in enumerate(si_range):
        chi2p_a[i], (norm_mu_a[i], norm_std_a[i], norm_c_a[i]) = fut_a[i].result()
        chi2p_b[i], (norm_mu_b[i], norm_std_b[i], norm_c_b[i]) = fut_b[i].result()

    # Plot
    chi2p_mean_a = np.nanmean(chi2p_a, axis=1)
    chi2p_std_a = np.nanstd(chi2p_a, axis=1)

    chi2p_mean_b = np.nanmean(chi2p_b, axis=1)
    chi2p_std_b = np.nanstd(chi2p_b, axis=1)

    norm_mu_mean_a = norm_mu_a.mean(axis=1)
    norm_mu_mean_b = norm_mu_b.mean(axis=1)

    norm_std_mean_a = norm_std_a.mean(axis=1)
    norm_std_mean_b = norm_std_b.mean(axis=1)

    const_dist_std = np.array([0.5-np.sqrt(1/12), 0.5+np.sqrt(1/12)])
    const_dist_std *= 100

    # RNG A
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), tight_layout=True)

    ax1.set_title('RNG A: PC Distrib Normal Fit, n_tests={}, n_pcs/test={}'
                  .format(get_ammount_prefix_str(n_tests), get_ammount_prefix_str(n_pcs)), loc='left')
    ax1.set_xlabel('SI')
    ax1.set_ylabel('p(%)')
    ax1.hlines([50], si_range.min()-0.5, si_range.max()+0.5, color='black', ls='solid', lw=0.6)
    ax1.hlines([const_dist_std], si_range.min()-0.5, si_range.max()+0.5, color='black', ls='dashed', lw=0.6)
    ax1.errorbar(si_range, chi2p_mean_a, chi2p_std_a, ls='none', color='black', marker='o', ms=4, capsize=4)
    ax1.set_ylim(-5, 100)
    ax1.set_xticks(si_range)
    ax1.set_yticks([0, 25, 50, 75, 100])

    ax2.set_xlabel('SI')
    ax2.set_ylabel('mean mu')
    lns1 = ax2.plot(si_range, norm_mu_mean_a, ls='none', color='black', ms=4, marker='o', fillstyle='none', label='mean mu')
    ax2.grid(axis='y')
    ax2.set_xticks(si_range)

    ax2t = ax2.twinx()
    ax2t.set_ylabel('mean sig')
    lns2 = ax2t.plot(si_range+0.25, norm_std_mean_a, ls='none', color='black', ms=4, marker='D', fillstyle='none', label='mean sig')

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, framealpha=0.8)

    # RNG B
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(7, 6), tight_layout=True)

    ax3.set_title('RNG B: PC Distrib Normal Fit, n_tests={}, n_pcs/test={}'
                  .format(get_ammount_prefix_str(n_tests), get_ammount_prefix_str(n_pcs)), loc='left')
    ax3.set_xlabel('SI')
    ax3.set_ylabel('p(%)')
    ax3.hlines([50], si_range.min()-0.5, si_range.max()+0.5, color='black', ls='solid', lw=0.6)
    ax3.hlines([const_dist_std], si_range.min()-0.5, si_range.max()+0.5, color='black', ls='dashed', lw=0.6)
    ax3.errorbar(si_range, chi2p_mean_b, chi2p_std_b, ls='none', color='gray', marker='o', ms=4, capsize=4)
    ax3.set_ylim(-5, 100)
    ax3.set_xticks(si_range)
    ax3.set_yticks([0, 25, 50, 75, 100])

    ax4.set_xlabel('SI')
    ax4.set_ylabel('mean mu')
    lns3 = ax4.plot(si_range, norm_mu_mean_b, ls='none', color='gray', ms=4,  marker='o', fillstyle='none', label='mean mu')
    ax4.grid(axis='y')
    ax4.set_xticks(si_range)

    ax4t = ax4.twinx()
    ax4t.set_ylabel('mean sig')
    lns4 = ax4t.plot(si_range+0.25, norm_std_mean_b, ls='none', color='gray', ms=4,  marker='D', fillstyle='none', label='mean sig')

    lns_b = lns3 + lns4
    labs_b = [l.get_label() for l in lns_b]
    ax4.legend(lns_b, labs_b, framealpha=0.8)

    return [fig1, fig2]