"""
Copyright (c) 2024 Gabriel Guerrer

Distributed under the MIT license - See LICENSE for details
"""

"""
This file encompasses the code for the reporting feature.
"""

import os.path
import tkinter.messagebox as tkm
import webbrowser

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from rng_rava.tk.acq import WIN_PROGRESS

import rng_rava_diag.tests_pcs as tests_pcs
import rng_rava_diag.tests_bytes as tests_bytes
import rng_rava_diag.tests_nist as tests_nist


def report(parent, path_report, file_pcs, file_bytes_a, file_bytes_b, file_nist_a, file_nist_b):
    ## Check files format
    if not 'PCS_50K_SI_10_10_1' in file_pcs:
        tkm.showerror(parent=parent, title='Pulse Count Error',
                        message='Provide a dataset acquired with the report preset parameters',
                        detail='Specifically: 50K Pulse Counts within the SI range of (10, 10, 1)')
        return

    if not 'BYTES_125M_PP0_A' in file_bytes_a:
        tkm.showerror(parent=parent, title='Bytes A Error',
                        message='Provide a dataset acquired with the report preset parameters',
                        detail='Specifically: 125M Bytes producey by the RNG A without any post-processing')
        return

    if not 'BYTES_125M_PP0_B' in file_bytes_b:
        tkm.showerror(parent=parent, title='Bytes B Error',
                        message='Provide a dataset acquired with the report preset parameters',
                        detail='Specifically: 125M Bytes producey by the RNG B without any post-processing')
        return

    ## Check Device SN
    sn_pcs = os.path.basename(file_pcs).split('_')[0]
    sn_bytes_a = os.path.basename(file_bytes_a).split('_')[0]
    sn_bytes_b = os.path.basename(file_bytes_b).split('_')[0]
    sns = [sn_pcs, sn_bytes_a, sn_bytes_b]

    if not all(sn == sns[0] for sn in sns):
        tkm.showerror(parent=parent, title='Report Error',
                        message='Provide datasets for the same Device',
                        detail='Conflicting Serial Numbers')
        return
    else:
        sn = sns[0]

    ## Progress
    win_progress = WIN_PROGRESS(parent)
    win_progress.show()
    n_tasks = 5

    ## Tests
    figs = []
    n_repeat = 5

    # Pulse Counts
    pcs_dict = np.load(file_pcs, allow_pickle=True)
    pcs_a = pcs_dict['pcs_a']
    pcs_b = pcs_dict['pcs_b']
    pwm_setup = pcs_dict['pwm_setup'][()]
    rng_setup = pcs_dict['rng_setup'][()]

    rng_setup_str = 'PWM freq={} duty={}, RNG SI={}us'.format(pwm_setup['freq_str'], pwm_setup['duty'], rng_setup['sampling_interval_us'])

    n_pcs_per_test = 10000
    pcs_2d_a = np.reshape(pcs_a, (n_repeat, n_pcs_per_test))
    pcs_2d_b = np.reshape(pcs_b, (n_repeat, n_pcs_per_test))

    for i in range(n_repeat):
        figs.append(tests_pcs.pcs_quick_test(pcs_2d_a[i], pcs_2d_b[i], rng_setup_str))

    win_progress.prog_update(1/n_tasks*100)

    # Bytes
    with open(file_bytes_a, 'br') as f_a:
        bytes_a = np.frombuffer(f_a.read(), dtype=np.uint8)

    with open(file_bytes_b, 'br') as f_b:
        bytes_b = np.frombuffer(f_b.read(), dtype=np.uint8)

    n_bytes_per_test = 125000
    n_tests_byte = len(bytes_a) // n_bytes_per_test
    bytes_2d_a = np.reshape(bytes_a, (n_tests_byte, n_bytes_per_test))
    bytes_2d_b = np.reshape(bytes_b, (n_tests_byte, n_bytes_per_test))

    win_progress.prog_update(2/n_tasks*100)

    # Bytes Quick
    for i in range(n_repeat):
        figs.append(tests_bytes.bytes_quick_test_bit_bias(bytes_2d_a[i], bytes_2d_b[i]))

    for i in range(n_repeat):
        figs.append(tests_bytes.bytes_quick_test_byte_bias(bytes_2d_a[i], bytes_2d_b[i]))

    win_progress.prog_update(3/n_tasks*100)

    # Bytes Detailed
    n_fit_bins = 20
    figs.extend(tests_bytes.bytes_detailed_test(bytes_2d_a, bytes_2d_b, n_fit_bins))

    win_progress.prog_update(4/n_tasks*100)

    # NIST
    if len(file_nist_a) and len(file_nist_b):
        figs.append(tests_nist.nist_test(file_nist_a, file_nist_b))

    win_progress.prog_update(5/n_tasks*100)

    ## Save PDF
    filename_pdf = os.path.join(path_report, '{}_report.pdf'.format(sn))

    with PdfPages(filename_pdf) as pdf:
        for fig in figs:
            pdf.savefig(fig)

    ## Close progress window
    win_progress.destroy()

    ## Show PDF
    webbrowser.open(filename_pdf)