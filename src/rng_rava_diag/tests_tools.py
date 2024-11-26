"""
Copyright (c) 2024 Gabriel Guerrer

Distributed under the MIT license - See LICENSE for details
"""

"""
Functions utilized by the Diagnostics app.
"""

import numpy as np
from scipy.stats import norm, chi2, binom, chisquare

from lmfit.models import Model
from lmfit import Parameters


## IO

def file_to_byte_array(file_name):
    with open(file_name, 'br') as f:
        byte_array = np.frombuffer(f.read(), dtype=np.uint8)
    return byte_array


## DISTRIBUTIONS

def normal_dist(x, mu, sig, c):
    dist = norm.pdf(x, loc=mu, scale=sig)
    return c * dist / dist.sum()


def chi2_dist(x, dof, c):
    dist = chi2.pdf(x, dof)
    return c * dist / dist.sum()


def binom_interval(interval, binom_n, binom_p):
    binom_dist = binom(n=binom_n, p=binom_p)
    ci_min, ci_max = binom_dist.interval(interval) # [min, max)
    ci_min = int(ci_min)
    ci_max = int(ci_max)
    interval_actual = binom_dist.pmf(np.arange(ci_min, ci_max)).sum()
    ci = [ci_min, ci_max - 1] # [min, max]
    return ci, interval_actual


## FITTING

def fit_normal(values, n_bins=None, debug=False):
    # Find histogram's max, min (and n_bins if values is of int type)
    NSIG = 4.0
    values_max = values.mean() + NSIG * values.std()
    values_min = values.mean() - NSIG * values.std()
    if np.issubdtype(values.dtype, np.integer):
        values_max = int(np.ceil(values_max)) + 1
        values_min = int(np.floor(values_min))
        n_bins = values_max - values_min

    # Histogram
    hist_values, hist_bins = np.histogram(values, n_bins, (values_min, values_max))
    hist_bins_mid = hist_bins[:-1] + (hist_bins[1] - hist_bins[0]) / 2

    # Fit parameters
    hist_area = hist_values.sum()
    pars0 = Parameters()
    pars0.add('mu', values.mean())
    pars0.add('sig', values.std())
    pars0.add('c', hist_area - hist_area * 0.1)

    # Errors
    # errs = np.sqrt(hist_values) # Poisson distrib
    errs = np.sqrt(hist_values * (1 - hist_values / hist_area)) # Normal approximation to binomial distrib
    errs[np.where(errs == 0)] = 1 # Avoid division by 0 error

    # Fit pulse counts to normal distrib
    normal_model = Model(normal_dist)
    fit_result = normal_model.fit(hist_values, pars0, x=hist_bins_mid, weights=1./errs)

    if debug:
        print(fit_result.fit_report())
        fit_result.plot_fit()

    # Extract parameters
    chisqr = fit_result.chisqr
    ndof = len(hist_values) - 3  # 3 fit parameters
    chi2p = chi2.sf(chisqr, ndof) * 100
    norm_mu = fit_result.params['mu'].value
    norm_mu_stderr = fit_result.params['mu'].stderr
    norm_std = fit_result.params['sig'].value
    norm_std_stderr = fit_result.params['sig'].stderr
    norm_c = fit_result.params['c'].value
    norm_c_stderr = fit_result.params['c'].stderr

    # Fit result histogram
    hist_fit_values = normal_dist(hist_bins_mid, norm_mu, norm_std, norm_c)
    hist_fit_values_err = np.sqrt( hist_fit_values * (1 - hist_fit_values / hist_fit_values.sum()) )

    return {'chi2p':chi2p,
            'fit_result':fit_result,
            'fit_pars':(norm_mu, norm_std, norm_c),
            'fit_pars_stderr':(norm_mu_stderr, norm_std_stderr, norm_c_stderr),
            'hist_values':(hist_values, hist_fit_values, hist_fit_values_err, hist_bins, hist_bins_mid)}


def fit_chi2(values, dof, n_bins=20):
    # Histogram
    chi2_min = chi2.isf(.9999+.0001/2, dof)
    chi2_max = chi2.isf(.0001/2, dof)
    hist_values, hist_bins = np.histogram(values, n_bins, (chi2_min, chi2_max))
    hist_bins_mid = hist_bins[:-1] + (hist_bins[1] - hist_bins[0]) / 2

    # Fit parameters
    hist_area = hist_values.sum()
    pars0 = Parameters()
    pars0.add('dof', dof - 3)
    pars0.add('c', hist_area - hist_area * 0.1)

    # Errors
    errs = np.sqrt(hist_values * (1 - hist_values / hist_area)) # Normal approximation to binomial distrib
    errs[np.where(errs == 0)] = 1 # Avoid division by 0 error

    # Fit pulse counts to normal distrib
    normal_model = Model(chi2_dist)
    fit_result = normal_model.fit(hist_values, pars0, x=hist_bins_mid, weights=1./errs)
    # print(fit_result.fit_report())
    # fit_result.plot_fit()

    # Extract parameters
    chisqr = fit_result.chisqr
    ndof = len(hist_values) - 2  # 2 fit parameters
    chi2p = chi2.sf(chisqr, ndof) * 100
    chi2_dof = fit_result.params['dof'].value
    chi2_dof_stderr = fit_result.params['dof'].stderr
    chi2_c = fit_result.params['c'].value
    chi2_c_stderr = fit_result.params['c'].stderr

    # Fit result histogram
    hist_fit_values = chi2_dist(hist_bins_mid, chi2_dof, chi2_c)
    hist_fit_values_err = np.sqrt( hist_fit_values * (1 - hist_fit_values / hist_fit_values.sum()) )

    return {'chi2p':chi2p,
            'fit_result':fit_result,
            'fit_pars':(chi2_dof, chi2_c),
            'fit_pars_stderr':(chi2_dof_stderr, chi2_c_stderr),
            'hist_values':(hist_values, hist_fit_values, hist_fit_values_err, hist_bins, hist_bins_mid)}


## STATISTICS

# Linked table that tracks the count of ones in all possible 256 bytes
popcount_lt = np.array([bin(i).count('1') for i in range(256)]).astype(np.uint8)


def bit_bias(byte_array):
    n_1s = popcount_lt[byte_array].sum()
    n_bits = 8*len(byte_array)
    bias = n_1s/n_bits - 0.5

    # Two-tailed p-value
    n_0s = n_bits - n_1s
    p_hi = binom.sf(max(n_0s, n_1s), n=n_bits, p=0.5)
    p_lo = binom.cdf(min(n_0s, n_1s), n=n_bits, p=0.5)
    p = p_lo + p_hi
    return bias, p


def bit_bias_to_z(bias, n_bits):
    return 2*bias*np.sqrt(n_bits)


def z_to_bit_bias(z, n_bits):
    return z/(2*np.sqrt(n_bits))


def theoretical_bit_bias(norm_mu, norm_std):
    norm_distrib = norm(norm_mu, norm_std)

    N_STDS = 15
    rnge_min = int(norm_mu - norm_std * N_STDS)
    rnge_max = int(norm_mu + norm_std * N_STDS)
    rnge = range(max(rnge_min, 0), rnge_max)

    if rnge[0] % 2:
        odd_idx_0 = 0
        even_idx_0 = 1
    else:
        odd_idx_0 = 1
        even_idx_0 = 0

    norm_p = norm_distrib.pdf(rnge)
    norm_p_sum = norm_p.sum()
    norm_p_even = norm_p[even_idx_0::2].sum() / norm_p_sum
    bias = np.abs(norm_p_even - .5)
    return bias


def byte_bias(byte_array):
    n_bytes = byte_array.size
    p = 1/256
    y, _ = np.histogram(byte_array, 256, range=(0, 256))
    chisq, p = chisquare(y, n_bytes*p)
    return chisq, p


def byte_bias_equivalent(byte_array):
    # Slower, but equivalent to byte_bias()
    expect = len(byte_array)/256
    ys, xs = np.histogram(byte_array, 256, range=(0, 256))
    chisq = 0
    for y in ys:
        chisq += (y - expect)**2 / expect
    p = chi2.sf(chisq, 255)
    return chisq, p


def nums_bias(nums_array, n_bins=100, bins_range=(0,1)):
    n_nums = nums_array.size
    p = 1/n_bins
    y, _ = np.histogram(nums_array, n_bins, range=bins_range)
    chisq, p = chisquare(y, n_nums*p)
    return chisq, p


def byte_to_bits(byte_array):
    return np.unpackbits(byte_array)


def serial_correl(byte_array, lag=1):
    bit_array = byte_to_bits(byte_array)
    if lag == 0:
        coef = 1.
    else:
        coef = np.corrcoef(bit_array[lag:], bit_array[:-lag])[0][1]
    return coef


def serial_correl_equivalent(byte_array, lag=1):
    # Slower, but equivalent to serial_correl()
    n_bits = 8 * len(byte_array)
    bits = byte_to_bits(byte_array)
    bits_lag = np.roll(bits, -lag)

    sc1 = np.sum(bits * bits_lag)
    sc2 = np.sum(bits) ** 2
    sc3 = np.sum(bits * bits)

    denom = n_bits * sc3 - sc2
    if denom == 0:
        raise ValueError
    num = (n_bits * sc1 - sc2)
    return num / denom


def correl_2arrays(byte_array1, byte_array2):
    bit_array1 = byte_to_bits(byte_array1)
    bit_array2 = byte_to_bits(byte_array2)
    coef = np.corrcoef(bit_array1, bit_array2)[0][1]
    return coef