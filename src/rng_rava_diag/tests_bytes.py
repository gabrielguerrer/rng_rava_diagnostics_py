"""
Copyright (c) 2024 Gabriel Guerrer

Distributed under the MIT license - See LICENSE for details
"""

"""
This file encompasses the code of the Bytes statistical tests performed
by the Quick and Detailed Tests sub-apps.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from rng_rava.acq import get_ammount_prefix_str

import rng_rava_diag.tests_tools as rt


## TESTS

def bytes_quick_test_bit_bias(bytes_a, bytes_b):
    # Calculate
    n_bits = len(bytes_a) * 8

    bias_a, p_a = rt.bit_bias(bytes_a)
    bias_b, p_b = rt.bit_bias(bytes_b)
    bias_a = bias_a * 100
    bias_b = bias_b * 100

    lbl_a = 'A: b={:.3f}%; p={:.1f}%'.format(bias_a, p_a*100)
    lbl_b = 'B: b={:.3f}%; p={:.1f}%'.format(bias_b, p_b*100)

    # 95% confidence interval
    ci95, interval = rt.binom_interval(.95, binom_n=n_bits, binom_p=0.5)
    ci95 = [(ci95[0]/n_bits - 0.5)*100, (ci95[1]/n_bits - 0.5)*100]
    lbl_bounds = '{:.1f}% CI'.format(interval*100)

    # Plot
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(5.5, 4))
    ax.set_title('Bit Bias, n_bits={}'.format(get_ammount_prefix_str(n_bits)), loc='left')
    ax.set_ylabel('Bias (%)')
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.plot([0], bias_a, ls='none', marker='o', color='black', label=lbl_a)
    ax.plot([0], bias_b, ls='none', marker='o', color='gray', label=lbl_b)
    ax.hlines(ci95, -0.4, 0.4, ls='dashed', lw=1, color='black', label=lbl_bounds)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 0.9))

    return fig


def bytes_quick_test_byte_bias(bytes_a, bytes_b):
    # Calculate
    n_bytes = len(bytes_a)

    n_categ = 256
    n_dof = n_categ - 1
    x = np.arange(256)
    hist_a, _ = np.histogram(bytes_a, n_categ, range=(0, n_categ))
    hist_b, _ = np.histogram(bytes_b, n_categ, range=(0, n_categ))

    chi2_a, p_a = rt.byte_bias(bytes_a)
    chi2_b, p_b = rt.byte_bias(bytes_b)

    lbl_a = 'A: chi2={:.2f}  p={:.1f}%'.format(chi2_a, p_a*100)
    lbl_b = 'B: chi2={:.2f}  p={:.1f}%'.format(chi2_b, p_b*100)

    # 95% confidence interval
    ci95, interval = rt.binom_interval(.95, binom_n=n_bytes, binom_p=1/n_categ)
    lbl_bounds = '{:.1f}% CI'.format(interval*100)

    n_exceeding_a = (hist_a < ci95[0]).sum() + (hist_a > ci95[1]).sum()
    n_exceeding_b = (hist_b < ci95[0]).sum() + (hist_b > ci95[1]).sum()
    lbl2_a = 'A: exceed={:.1f}%'.format(n_exceeding_a/n_categ*100)
    lbl2_b = 'B: exceed={:.1f}%'.format(n_exceeding_b/n_categ*100)

    # Plot
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(7, 5))
    ax.set_title('Byte bias, n_bytes={}, dof={}'.format(get_ammount_prefix_str(n_bytes), n_dof), loc='left')
    ax.set_xlabel('Categories')
    ax.set_ylabel('N')
    ax.plot(x, hist_a, ls='none', marker='o', ms=3, color='black', label=lbl_a)
    ax.plot(x, hist_b, ls='none', marker='o', ms=3, color='gray', label=lbl_b)
    ax.hlines(ci95, 0, 255, ls='dashed', lw=1, color='black', label=lbl_bounds)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    ax2 = ax.twinx()
    ax2.plot(np.nan, np.nan, ls='none', marker='o', ms=3, color='black', label=lbl2_a)
    ax2.plot(np.nan, np.nan, ls='none', marker='o', ms=3, color='gray', label=lbl2_b)
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc='lower left', bbox_to_anchor=(0, 0))

    return fig


def bytes_quick_test_correls(bytes_a, bytes_b):
    # Calculate
    n_bits = len(bytes_a) * 8

    sc_a = rt.serial_correl(bytes_a) * 100
    sc_b = rt.serial_correl(bytes_b) * 100
    c_ab = rt.correl_2arrays(bytes_a, bytes_b) * 100

    lbl_a = 'A: SC={:.4f}%'.format(sc_a)
    lbl_b = 'B: SC={:.4f}%'.format(sc_b)
    lbl_c = 'AB: CORR={:.4f}%'.format(c_ab)

    # Plot
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(5.5, 4))
    ax.set_title('Correlation, n_bits={}'.format(get_ammount_prefix_str(n_bits)), loc='left')
    ax.set_ylabel('Correl (%)')
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.plot([0], sc_a, ls='none', marker='o', color='black', label=lbl_a)
    ax.plot([0], sc_b, ls='none', marker='o', color='gray', label=lbl_b)
    ax.plot([0], c_ab, ls='none', marker='s', color='darkgray', label=lbl_c)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

    return fig


def bytes_calc(bytes):
    # Load data
    n_tests = bytes.shape[0]

    # Results arrays
    bias = np.zeros(n_tests, dtype=np.float64)
    chi2 = np.zeros(n_tests, dtype=np.float64)
    sc = np.zeros(n_tests, dtype=np.float64)

    # Loop and calculate
    for i in range(n_tests):
        bias[i], _ = rt.bit_bias(bytes[i])
        bias[i] *= 100
        chi2[i], _ = rt.byte_bias(bytes[i])
        sc[i] = rt.serial_correl(bytes[i], 1) * 100

    return bias, chi2, sc


def bytes_calc_corr(bytes_a, bytes_b):
    # Load data
    n_tests = bytes_a.shape[0]

    # Results arrays
    corr = np.zeros(n_tests, dtype=np.float64)

    # Loop and calculate
    for i in range(n_tests):
        corr[i] = rt.correl_2arrays(bytes_a[i], bytes_b[i]) * 100

    return corr


def bytes_detailed_test(bytes_2d_a, bytes_2d_b, n_fit_bins):
    # Calc vars
    n_tests = bytes_2d_a.shape[0]
    n_bytes = bytes_2d_a.shape[1]

    pool = ProcessPoolExecutor()
    fut_a = pool.submit(bytes_calc, bytes_2d_a)
    fut_b = pool.submit(bytes_calc, bytes_2d_b)
    fut_c = pool.submit(bytes_calc_corr, bytes_2d_a, bytes_2d_b)

    bias_a, chi2_a, sc_a = fut_a.result()
    bias_b, chi2_b, sc_b = fut_b.result()
    corr_ab = fut_c.result()

    #########################
    # Bit bias

    # RNG A
    bias_dict_a = rt.fit_normal(bias_a, n_bins=n_fit_bins)
    bias_fit_chi2p_a = bias_dict_a['chi2p']
    bias_fit_mu_a, bias_fit_std_a, bias_fit_c_a = bias_dict_a['fit_pars']
    bias_fit_mu_err_a, bias_fit_std_err_a, bias_fit_c_err_a = bias_dict_a['fit_pars_stderr']
    bias_hist_a, bias_hist_fit_a, bias_err_fit_a, bias_bins_a, bias_bins_mid_a = bias_dict_a['hist_values']

    lbl_bias_a = 'A: mu={:.4f} +- {:.4f}; std={:.4f} +- {:.4f}; \nc={:.0f}+- {:.0f} ; p={:.2f}%' \
                 .format(bias_fit_mu_a, 1.96*bias_fit_mu_err_a, bias_fit_std_a, 1.96 * bias_fit_std_err_a,
                    bias_fit_c_a, 1.96 * bias_fit_c_err_a, bias_fit_chi2p_a)

    # RNG B
    bias_dict_b = rt.fit_normal(bias_b, n_bins=n_fit_bins)
    bias_fit_chi2p_b = bias_dict_b['chi2p']
    bias_fit_mu_b, bias_fit_std_b, bias_fit_c_b = bias_dict_b['fit_pars']
    bias_fit_mu_err_b, bias_fit_std_err_b, bias_fit_c_err_b = bias_dict_b['fit_pars_stderr']
    bias_hist_b, bias_hist_fit_b, bias_err_fit_b, bias_bins_b, bias_bins_mid_b = bias_dict_b['hist_values']

    lbl_bias_b = 'B: mu={:.4f} +- {:.4f}; std={:.4f} +- {:.4f}; \nc={:.0f}+- {:.0f} ; p={:.2f}%'\
                 .format(bias_fit_mu_b, 1.96*bias_fit_mu_err_b, bias_fit_std_b, 1.96 * bias_fit_std_err_b,
                    bias_fit_c_b, 1.96 * bias_fit_c_err_b, bias_fit_chi2p_b)

    #########################
    ## Byte bias

    byte_ndof = 255

    # RNG A
    chi2_dict_a = rt.fit_chi2(chi2_a, dof=byte_ndof, n_bins=n_fit_bins)
    chi2_fit_chi2p_a = chi2_dict_a['chi2p']
    chi2_fit_dof_a, chi2_fit_c_a = chi2_dict_a['fit_pars']
    chi2_fit_dof_err_a, chi2_fit_c_err_a = chi2_dict_a['fit_pars_stderr']
    chi2_hist_a, chi2_hist_fit_a, chi2_err_fit_a, chi2_bins_a, chi2_bins_mid_a = chi2_dict_a['hist_values']

    lbl_chi2_a = 'A: dof={:.2f} +- {:.2f}; \nc={:.0f} +- {:.0f}; p={:.2f}%' \
            .format(chi2_fit_dof_a, 1.96*chi2_fit_dof_err_a, chi2_fit_c_a, 1.96 * chi2_fit_c_err_a, chi2_fit_chi2p_a)

    # RNG B
    chi2_dict_b = rt.fit_chi2(chi2_b, dof=byte_ndof, n_bins=n_fit_bins)
    chi2_fit_chi2p_b = chi2_dict_b['chi2p']
    chi2_fit_dof_b, chi2_fit_c_b = chi2_dict_b['fit_pars']
    chi2_fit_dof_err_b, chi2_fit_c_err_b = chi2_dict_b['fit_pars_stderr']
    chi2_hist_b, chi2_hist_fit_b, chi2_err_fit_b, chi2_bins_b, chi2_bins_mid_b = chi2_dict_b['hist_values']

    lbl_chi2_b = 'B: dof={:.2f} +- {:.2f}; \nc={:.0f} +- {:.0f}; p={:.2f}%' \
            .format(chi2_fit_dof_b, 1.96*chi2_fit_dof_err_b, chi2_fit_c_b, 1.96 * chi2_fit_c_err_b, chi2_fit_chi2p_b)


    # #########################
    # Correlation

    # RNG A
    sc_dict_a = rt.fit_normal(sc_a, n_bins=n_fit_bins)
    sc_fit_chi2p_a = sc_dict_a['chi2p']
    sc_fit_mu_a, sc_fit_std_a, sc_fit_c_a = sc_dict_a['fit_pars']
    sc_fit_mu_err_a, sc_fit_std_err_a, sc_fit_c_err_a = sc_dict_a['fit_pars_stderr']
    sc_hist_a, sc_hist_fit_a, sc_err_fit_a, sc_bins_a, sc_bins_mid_a = sc_dict_a['hist_values']

    lbl_sc_a = 'A: mu={:.4f} +- {:.4f}; std={:.4f} +- {:.4f}; \nc={:.0f}+- {:.0f} ; p={:.2f}%' \
                 .format(sc_fit_mu_a, 1.96*sc_fit_mu_err_a, sc_fit_std_a, 1.96 * sc_fit_std_err_a,
                    sc_fit_c_a, 1.96 * sc_fit_c_err_a, sc_fit_chi2p_a)

    # RNG B
    sc_dict_b = rt.fit_normal(sc_b, n_bins=n_fit_bins)
    sc_fit_chi2p_b = sc_dict_b['chi2p']
    sc_fit_mu_b, sc_fit_std_b, sc_fit_c_b = sc_dict_b['fit_pars']
    sc_fit_mu_err_b, sc_fit_std_err_b, sc_fit_c_err_b = sc_dict_b['fit_pars_stderr']
    sc_hist_b, sc_hist_fit_b, sc_err_fit_b, sc_bins_b, sc_bins_mid_b = sc_dict_b['hist_values']

    lbl_sc_b = 'B: mu={:.4f} +- {:.4f}; std={:.4f} +- {:.4f}; \nc={:.0f}+- {:.0f} ; p={:.2f}%' \
                 .format(sc_fit_mu_b, 1.96*sc_fit_mu_err_b, sc_fit_std_b, 1.96 * sc_fit_std_err_b,
                    sc_fit_c_b, 1.96 * sc_fit_c_err_b, sc_fit_chi2p_b)

    # RNG A vs RNG B
    corr_dict = rt.fit_normal(corr_ab, n_bins=n_fit_bins)
    corr_fit_chi2p = corr_dict['chi2p']
    corr_fit_mu, corr_fit_std, corr_fit_c = corr_dict['fit_pars']
    corr_fit_mu_err, corr_fit_std_err, corr_fit_c_err = corr_dict['fit_pars_stderr']
    corr_hist, corr_hist_fit, corr_err_fit, corr_bins, corr_bins_mid = corr_dict['hist_values']

    lbl_corr = 'AxB: mu={:.4f} +- {:.4f}; std={:.4f} +- {:.4f}; \nc={:.0f}+- {:.0f} ; p={:.2f}%' \
                  .format(corr_fit_mu, 1.96*corr_fit_mu_err, corr_fit_std, 1.96 * corr_fit_std_err,
                    corr_fit_c, 1.96 * corr_fit_c_err, corr_fit_chi2p)

    #########################
    ## Plots

    # Bit Bias
    fig1, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(10, 4))
    ax1.set_title('Bit Bias Distribution, n_tests={}, n_bits/test={}'
                  .format(get_ammount_prefix_str(n_tests), get_ammount_prefix_str(n_bytes*8)), loc='left')

    ax1.set_ylabel('N')
    ax1.stairs(bias_hist_a, bias_bins_a, color='black', lw=1.5, label=lbl_bias_a)
    ax1.plot(bias_bins_mid_a, bias_hist_fit_a, color='black', lw=1, ls='--')
    ax1.errorbar(bias_bins_mid_a, bias_hist_fit_a, 1.96 * bias_err_fit_a, color='black', lw=1, ls='none', capsize=2, marker='o', ms=1)
    ymin_1, ymax_1 = ax1.get_ylim()
    ax1.set_ylim(ymin_1, 1.15 * ymax_1)
    ax1.legend(loc='upper right', bbox_to_anchor=(1., 1.))

    ax2.set_ylabel('N')
    ax2.stairs(bias_hist_b, bias_bins_b, color='gray', lw=1.5, label=lbl_bias_b)
    ax2.plot(bias_bins_mid_b, bias_hist_fit_b, color='gray', lw=1, ls='--')
    ax2.errorbar(bias_bins_mid_b, bias_hist_fit_b, 1.96 * bias_err_fit_b, color='gray', lw=1, ls='none', capsize=2, marker='o', ms=1)
    ymin_2, ymax_2 = ax2.get_ylim()
    ax2.set_ylim(ymin_2, 1.15 * ymax_2)
    ax2.legend(loc='upper right', bbox_to_anchor=(1., 1.))

    # Byte Bias
    fig2, (ax3, ax4) = plt.subplots(1, 2, tight_layout=True, figsize=(10, 4))
    ax3.set_title('Byte Bias Distribution, n_tests={}, n_bytes/test={}'
                  .format(get_ammount_prefix_str(n_tests), get_ammount_prefix_str(n_bytes)), loc='left')

    ax3.set_ylabel('N')
    ax3.stairs(chi2_hist_a, chi2_bins_a, color='black', lw=1.5, label=lbl_chi2_a)
    ax3.plot(chi2_bins_mid_a, chi2_hist_fit_a, color='black', lw=1, ls='--')
    ax3.errorbar(chi2_bins_mid_a, chi2_hist_fit_a, 1.96 * chi2_err_fit_a, color='black', lw=1, ls='none', capsize=2, marker='o', ms=1)
    ymin_3, ymax_3 = ax3.get_ylim()
    ax3.set_ylim(ymin_3, 1.15 * ymax_3)
    ax3.legend(loc='upper right', bbox_to_anchor=(1., 1.))

    ax4.set_ylabel('N')
    ax4.stairs(chi2_hist_b, chi2_bins_b, color='gray', lw=1.5, label=lbl_chi2_b)
    ax4.plot(chi2_bins_mid_b, chi2_hist_fit_b, color='gray', lw=1, ls='--')
    ax4.errorbar(chi2_bins_mid_b, chi2_hist_fit_b, 1.96 * chi2_err_fit_b, color='gray', lw=1, ls='none', capsize=2, marker='o', ms=1)
    ymin_4, ymax_4 = ax4.get_ylim()
    ax4.set_ylim(ymin_4, 1.15 * ymax_4)
    ax4.legend(loc='upper right', bbox_to_anchor=(1., 1.))

    # Correl
    fig3, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, tight_layout=True, figsize=(10, 7))
    ax5.set_title('Correlation Distribution, n_tests={},  n_bits/test={}'
                  .format(get_ammount_prefix_str(n_tests), get_ammount_prefix_str(n_bytes*8)), loc='left')

    ax5.set_ylabel('N')
    ax5.stairs(sc_hist_a, sc_bins_a, color='black', lw=1.5, label=lbl_sc_a)
    ax5.plot(sc_bins_mid_a, sc_hist_fit_a, color='black', lw=1, ls='--')
    ax5.errorbar(sc_bins_mid_a, sc_hist_fit_a, 1.96 * sc_err_fit_a, color='black', lw=1, ls='none', capsize=2, marker='o', ms=1)
    ymin_5, ymax_5 = ax5.get_ylim()
    ax5.set_ylim(ymin_5, 1.2 * ymax_5)
    ax5.legend(loc='upper right', bbox_to_anchor=(1., 1.))

    ax6.set_ylabel('N')
    ax6.stairs(sc_hist_b, sc_bins_b, color='gray', lw=1.5, label=lbl_sc_b)
    ax6.plot(sc_bins_mid_b, sc_hist_fit_b, color='gray', lw=1, ls='--')
    ax6.errorbar(sc_bins_mid_b, sc_hist_fit_b, 1.96 * sc_err_fit_b, color='gray', lw=1, ls='none', capsize=2, marker='o', ms=1)
    ymin_6, ymax_6 = ax6.get_ylim()
    ax6.set_ylim(ymin_6, 1.2 * ymax_6)
    ax6.legend(loc='upper right', bbox_to_anchor=(1., 1.))

    ax7.set_ylabel('N')
    ax7.stairs(corr_hist, corr_bins, color='darkgray', lw=1.5, label=lbl_corr)
    ax7.plot(corr_bins_mid, corr_hist_fit, color='darkgray', lw=1, ls='--')
    ax7.errorbar(corr_bins_mid, corr_hist_fit, 1.96 * corr_err_fit, color='gray', lw=1, ls='none', capsize=2, marker='o', ms=1)
    ymin_7, ymax_7 = ax7.get_ylim()
    ax7.set_ylim(ymin_7, 1.2 * ymax_7)
    ax7.legend(loc='upper right', bbox_to_anchor=(1., 1.))

    ax8.set_axis_off()

    return [fig1, fig2, fig3]