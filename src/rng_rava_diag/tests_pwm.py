"""
Copyright (c) 2024 Gabriel Guerrer

Distributed under the MIT license - See LICENSE for details
"""

"""
This file encompasses the code of the PWM tests performed by the Quick
Tests sub-app.
"""

import matplotlib.pyplot as plt


def pwm_test(pcs_a, pcs_b, sampling_interval, pwm_freq, pwm_dc_range):
    # Calculate
    pcs_m_a = pcs_a.mean(axis=1)
    pcs_m_b = pcs_b.mean(axis=1)

    # Plot
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 5))
    ax.grid(axis='y')
    ax.set_title('Pulse Count Evolution; PWM freq={}, RNG SI={}us'.format(pwm_freq, sampling_interval), loc='left')
    ax.set_xlabel('Duty Cycle')
    ax.set_ylabel('PC mean')
    ax.plot(pwm_dc_range, pcs_m_a, marker='o', color='black', label='A')
    ax.plot(pwm_dc_range, pcs_m_b, marker='o', color='gray', label='B')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    return fig