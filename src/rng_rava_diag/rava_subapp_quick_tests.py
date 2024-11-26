"""
Copyright (c) 2024 Gabriel Guerrer

Distributed under the MIT license - See LICENSE for details
"""

"""
The Quick Tests sub-app comprises statistical tests conducted on relatively
small samples generated in real-time to detect significant errors within the
RAVA circuit.

For a more comprehensive assessment, generate randomness files using the
Acquisition sub-app and subsequently analyze them using the Detailed Tests
sub-app.
"""

import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as tkm

import numpy as np
import matplotlib.pyplot as plt

from rng_rava import D_RNG_POSTPROC, D_PWM_FREQ
from rng_rava.tk import RAVA_SUBAPP
from rng_rava.tk.acq import WIN_PROGRESS
from rng_rava.acq import RAVA_ACQUISITION, get_ammount_prefix_number

import rng_rava_diag.tests_pcs as tests_pcs
import rng_rava_diag.tests_bytes as tests_bytes
import rng_rava_diag.tests_nums as tests_nums
import rng_rava_diag.tests_pwm as tests_pwm


### VARS

PAD = 10


### SUBAPP_TESTS

class RAVA_SUBAPP_QUICK_TESTS(RAVA_SUBAPP):

    cfg_default_str = ''

    def __init__(self, parent):
        ## Initialize RAVA_SUBAPP
        name = 'RAVA_SUBAPP QUICK TESTS'
        win_title = 'Quick Tests'
        win_geometry = '400x250'
        win_resizable = False
        if not super().__init__(parent, name=name, win_title=win_title, win_geometry=win_geometry, win_resizable=win_resizable):
            return

        # Windows
        self.win_progress = WIN_PROGRESS(self)
        self.win_progress.hide()

        # Widgets
        self.nb = ttk.Notebook(self, padding=PAD)
        self.nb.grid(row=0, column=0, sticky='nsew')

        self.frm_pcs = ttk.Frame(self, name='pulse_counts', padding=(0,PAD))
        self.frm_pcs.grid(row=0, column=0, sticky='nsew')
        self.pcs_widgets()

        self.frm_bytes = ttk.Frame(self, name='bytes', padding=(0,PAD))
        self.frm_bytes.grid(row=0, column=0, sticky='nsew')
        self.bytes_widgets()

        self.frm_nums = ttk.Frame(self, name='numbers', padding=(0,PAD))
        self.frm_nums.grid(row=0, column=0, sticky='nsew')
        self.nums_widgets()

        self.frm_pwm = ttk.Frame(self, name='pwm', padding=(0,PAD))
        self.frm_pwm.grid(row=0, column=0, sticky='nsew')
        self.pwm_widgets()

        self.nb.add(self.frm_pcs, text=' Pulse Counts ')
        self.nb.add(self.frm_bytes, text=' Bytes ')
        self.nb.add(self.frm_nums, text=' Numbers ')
        self.nb.add(self.frm_pwm, text=' PWM ')

        # Key binds
        self.bind('<Control-Key-c>', lambda event=None: self.plots_close())

        ## Start
        self.plots = []

        # RAVA_ACQUISITION
        self.rng_acq = RAVA_ACQUISITION(delete_incomplete=True)
        self.rng_acq.cbkreg_progress(self.win_progress.prog_update)


    def close(self):
        # Close open plots
        self.plots_close()

        # Close RAVA_SUBAPP
        super().close()


    def plots_close(self):
        for f in self.plots:
            plt.close(f)
            del f


    def pcs_widgets(self):
        self.frm_pcs.columnconfigure([0,1,2,3], weight=1)
        self.frm_pcs.rowconfigure([0,1], weight=1)

        ## Parameters
        self.lb_pcs_n = ttk.Label(self.frm_pcs, text='N PCs')
        self.lb_pcs_n.grid(row=0, column=0, sticky='w', padx=(2*PAD,0))

        self.var_pcs_n = tk.DoubleVar(value=10)
        self.spb_pcs_n = ttk.Spinbox(self.frm_pcs, from_=1, to=999, increment=1, textvariable=self.var_pcs_n, width=8)
        self.spb_pcs_n.grid(row=0, column=1, sticky='w')

        self.cbb_pcs_n_prefix = ttk.Combobox(self.frm_pcs, width=9)
        self.cbb_pcs_n_prefix.grid(row=0, column=2, sticky='w')
        self.cbb_pcs_n_prefix.state(['readonly'])
        self.cbb_pcs_n_prefix['values'] = ['', 'K', 'M', 'G', 'T']
        self.cbb_pcs_n_prefix.set('K')

        ## Tests
        self.bt_pcs_test = ttk.Button(self.frm_pcs, text='Test', width=16, command=self.pcs_test)
        self.bt_pcs_test.grid(row=1, column=0, columnspan=4)


    def bytes_widgets(self):
        self.frm_bytes.columnconfigure([0,1,2], weight=1)
        self.frm_bytes.rowconfigure([0,1,2,3,4], weight=1)

        ## Parameters
        self.lb_bytes_n = ttk.Label(self.frm_bytes, text='N Bytes')
        self.lb_bytes_n.grid(row=0, column=0, sticky='w', padx=(2*PAD,0))

        self.var_bytes_n = tk.DoubleVar(value=125)
        self.spb_bytes_n = ttk.Spinbox(self.frm_bytes, from_=1, to=999, increment=1, textvariable=self.var_bytes_n, width=8)
        self.spb_bytes_n.grid(row=0, column=1, sticky='w')

        self.cbb_bytes_n_prefix = ttk.Combobox(self.frm_bytes, width=9)
        self.cbb_bytes_n_prefix.grid(row=0, column=2, sticky='w')
        self.cbb_bytes_n_prefix.state(['readonly'])
        self.cbb_bytes_n_prefix['values'] = ['B', 'KB', 'MB', 'GB', 'TB']
        self.cbb_bytes_n_prefix.set('KB')

        self.lb_bytes_postproc = ttk.Label(self.frm_bytes, text='Post-proc')
        self.lb_bytes_postproc.grid(row=1, column=0, sticky='w', padx=(2*PAD,0))

        self.cbb_bytes_postproc = ttk.Combobox(self.frm_bytes, width=15)
        self.cbb_bytes_postproc.grid(row=1, column=1, columnspan=2, sticky='w')
        self.cbb_bytes_postproc.state(['readonly'])
        self.cbb_bytes_postproc['values'] = list(D_RNG_POSTPROC.keys())
        self.cbb_bytes_postproc.set('NONE')

        self.var_bytes_bit_bias = tk.BooleanVar(value=True)
        self.cb_bytes_bit_bias = ttk.Checkbutton(self.frm_bytes, text='Bit Bias', variable=self.var_bytes_bit_bias)
        self.cb_bytes_bit_bias.grid(row=2, column=0)

        self.var_bytes_byte_bias = tk.BooleanVar(value=True)
        self.cb_bytes_byte_bias = ttk.Checkbutton(self.frm_bytes, text='Byte Bias', variable=self.var_bytes_byte_bias)
        self.cb_bytes_byte_bias.grid(row=2, column=1)

        self.var_bytes_correl = tk.BooleanVar(value=False)
        self.cb_bytes_correl = ttk.Checkbutton(self.frm_bytes, text='Correlation', variable=self.var_bytes_correl)
        self.cb_bytes_correl.grid(row=2, column=2)

        ## Tests
        self.bt_bytes_test = ttk.Button(self.frm_bytes, text='Test', command=self.bytes_test)
        self.bt_bytes_test.grid(row=3, column=0, columnspan=3)


    def nums_widgets(self):
        self.frm_nums.columnconfigure([0,1,2,3], weight=1)
        self.frm_nums.rowconfigure([0,1,2,3,4], weight=1)

        ## Parameters

        # Nums type
        self.lb_nums_type = ttk.Label(self.frm_nums, text='Type')
        self.lb_nums_type.grid(row=0, column=0, sticky='w', padx=(2*PAD,0))

        self.cbb_nums_type = ttk.Combobox(self.frm_nums, width=9)
        self.cbb_nums_type.grid(row=0, column=1, sticky='w')
        self.cbb_nums_type.state(['readonly'])
        self.cbb_nums_type['values'] = ['INT', 'FLOAT']
        self.cbb_nums_type.set('INT')
        self.cbb_nums_type.bind('<<ComboboxSelected>>', self.nums_type_sel)

        # Nums N
        self.lb_nums_n = ttk.Label(self.frm_nums, text='N Nums')
        self.lb_nums_n.grid(row=1, column=0, sticky='w', padx=(2*PAD,0))

        self.var_nums_n = tk.DoubleVar(value=50)
        self.spb_nums_n = ttk.Spinbox(self.frm_nums, from_=1, to=999, increment=1, textvariable=self.var_nums_n, width=8)
        self.spb_nums_n.grid(row=1, column=1, sticky='w')

        self.cbb_nums_n_prefix = ttk.Combobox(self.frm_nums, width=9)
        self.cbb_nums_n_prefix.grid(row=1, column=2, sticky='w')
        self.cbb_nums_n_prefix.state(['readonly'])
        self.cbb_nums_n_prefix['values'] = ['', 'K', 'M', 'G', 'T']
        self.cbb_nums_n_prefix.set('K')

        # Nums range
        self.lb_nums_range = ttk.Label(self.frm_nums, text='Range')
        self.lb_nums_range.grid(row=2, column=0, sticky='w', padx=(2*PAD,0))

        self.var_nums_range_min = tk.DoubleVar(value=0)
        self.spb_nums_range_min = ttk.Spinbox(self.frm_nums, from_=1, to=999, increment=1, textvariable=self.var_nums_range_min, width=8)
        self.spb_nums_range_min.grid(row=2, column=1, sticky='w')

        self.var_nums_range_max = tk.DoubleVar(value=999)
        self.spb_nums_range_max = ttk.Spinbox(self.frm_nums, from_=1, to=999, increment=1, textvariable=self.var_nums_range_max, width=8)
        self.spb_nums_range_max.grid(row=2, column=2, sticky='w')

        # Float bins
        self.lb_nums_float_bins = ttk.Label(self.frm_nums, text='Float bins')
        self.lb_nums_float_bins.grid(row=3, column=0, sticky='w', padx=(2*PAD,0))

        self.var_nums_float_bins = tk.IntVar(value=1000)
        self.spb_nums_float_bins = ttk.Spinbox(self.frm_nums, from_=1, to=100000, increment=1, textvariable=self.var_nums_float_bins, width=8)
        self.spb_nums_float_bins.grid(row=3, column=1, sticky='w')
        self.spb_nums_float_bins.state(['disabled'])

        ## Tests
        self.bt_nums_test = ttk.Button(self.frm_nums, text='Test', command=self.nums_test)
        self.bt_nums_test.grid(row=4, column=0, columnspan=4)


    def pwm_widgets(self):
        self.frm_pwm.columnconfigure([0,1,2,3], weight=1)
        self.frm_pwm.rowconfigure([0,1,2,3,4], weight=1)

        ## Parameters
        # SI
        self.lb_pwm_si = ttk.Label(self.frm_pwm, text='RNG SI (us)')
        self.lb_pwm_si.grid(row=0, column=0, sticky='w', padx=(2*PAD,0))

        self.var_pwm_si = tk.IntVar(value=10)
        self.spb_pwm_si = ttk.Spinbox(self.frm_pwm, from_=1, to=50, increment=1, textvariable=self.var_pwm_si, width=9)
        self.spb_pwm_si.grid(row=0, column=1, columnspan=2, sticky='w')

        # PWM Freq
        self.lb_pwm_freq = ttk.Label(self.frm_pwm, text='PWM Frequency')
        self.lb_pwm_freq.grid(row=1, column=0, sticky='w', padx=(2*PAD,0))

        pwm_freqs = list(D_PWM_FREQ.keys())
        self.cbb_pwm_freq = ttk.Combobox(self.frm_pwm, width=10)
        self.cbb_pwm_freq.grid(row=1, column=1, columnspan=2, sticky='w')
        self.cbb_pwm_freq.state(['readonly'])
        self.cbb_pwm_freq['values'] = pwm_freqs
        self.cbb_pwm_freq.set(pwm_freqs[2])
        self.cbb_pwm_freq.bind('<<ComboboxSelected>>', self.pwm_freq_sel)

        # PWM Duty Cycle
        self.lb_pwm_dc = ttk.Label(self.frm_pwm, text='PWM DC range\n(min, max, step)')
        self.lb_pwm_dc.grid(row=2, column=0, sticky='w', padx=(2*PAD,0))

        self.var_pwm_dc_min = tk.IntVar(value=4)
        self.spb_pwm_dc_min = ttk.Spinbox(self.frm_pwm, from_=1, to=50, increment=1, textvariable=self.var_pwm_dc_min, width=5)
        self.spb_pwm_dc_min.grid(row=2, column=1, sticky='w')

        self.var_pwm_dc_max = tk.IntVar(value=40)
        self.spb_pwm_dc_max = ttk.Spinbox(self.frm_pwm, from_=1, to=50, increment=1, textvariable=self.var_pwm_dc_max, width=5)
        self.spb_pwm_dc_max.grid(row=2, column=2, sticky='w')

        self.var_pwm_dc_step = tk.IntVar(value=2)
        self.spb_pwm_dc_step = ttk.Spinbox(self.frm_pwm, from_=1, to=50, increment=1, textvariable=self.var_pwm_dc_step, width=5)
        self.spb_pwm_dc_step.grid(row=2, column=3, sticky='w')

        # N PCs
        self.lb_pwm_pcs_n = ttk.Label(self.frm_pwm, text='N PCs per DC')
        self.lb_pwm_pcs_n.grid(row=3, column=0, sticky='w', padx=(2*PAD,0))

        self.var_pwm_pcs_n = tk.DoubleVar(value=10)
        self.spb_pwm_pcs_n = ttk.Spinbox(self.frm_pwm, from_=1, to=999, increment=1, textvariable=self.var_pwm_pcs_n, width=5)
        self.spb_pwm_pcs_n.grid(row=3, column=1, sticky='w')

        self.cbb_pwm_pcs_n_prefix = ttk.Combobox(self.frm_pwm, width=6)
        self.cbb_pwm_pcs_n_prefix.grid(row=3, column=2, sticky='w')
        self.cbb_pwm_pcs_n_prefix.state(['readonly'])
        self.cbb_pwm_pcs_n_prefix['values'] = ['B', 'K', 'M', 'G', 'T']
        self.cbb_pwm_pcs_n_prefix.set('K')

        ## Tests
        self.bt_pwm_test = ttk.Button(self.frm_pwm, text='Test', command=self.pwm_test)
        self.bt_pwm_test.grid(row=4, column=0, columnspan=4)


    def pcs_test(self):
        # Read vars
        n_pcs = int(get_ammount_prefix_number(n=self.var_pcs_n.get(), prefix=self.cbb_pcs_n_prefix.get()))
        rng_out = 'AB'
        n_chunk = self.cfg.read('ACQUISITION', 'chunk_bytes', int)

        # Info
        self.lg.info('{}: Pulse Count - Distribution Test'.format(self.name))

        # Get RNG setup str
        pwm_setup = self.rng.get_pwm_setup()
        rng_setup = self.rng.get_rng_setup()
        rng_setup_str = 'PWM freq={} duty={}, RNG SI={}us'.format(pwm_setup['freq_str'], pwm_setup['duty'], rng_setup['sampling_interval_us'])

        # Show win progress
        self.win_progress.show()

        # Acquire
        outputs, progress, time_str = self.rng_acq.get_pulse_counts(rng=self.rng, n_pcs=n_pcs, n_chunk=n_chunk,
                                        rng_out=rng_out, out_file=False, threaded=False)

        # Hide win progress
        self.win_progress.hide()

        # Test outputs
        if progress == 0:
            tkm.showerror(parent=self, title='Error', message='Pulse Count acquisition failed')
            return
        elif progress < 100: # Canceled
            return

        # Perform tests
        pcs_a, pcs_b = outputs

        fig = tests_pcs.pcs_quick_test(pcs_a, pcs_b, rng_setup_str)
        fig.show()
        self.plots.append(fig)


    def bytes_test(self):        
        get_ammount_prefix_number(n=self.var_bytes_n.get(), prefix=self.cbb_bytes_n_prefix.get())
        # Read vars
        n_bytes = int(get_ammount_prefix_number(n=self.var_bytes_n.get(), prefix=self.cbb_bytes_n_prefix.get()))
        postproc = self.cbb_bytes_postproc.get()
        rng_out = 'AB'
        n_chunk = self.cfg.read('ACQUISITION', 'chunk_bytes', int)

        # Info
        self.lg.info('{}: Bytes - Bias/Correl Tests'.format(self.name))

        # Show win progress
        self.win_progress.show()

        # Acquire
        outputs, progress, time_str = self.rng_acq.get_bytes(rng=self.rng, n_bytes=n_bytes, n_chunk=n_chunk,
                                        postproc=postproc, rng_out=rng_out, out_file=False, threaded=False)

        # Hide win progress
        self.win_progress.hide()

        # Test outputs
        if progress == 0:
            tkm.showerror(parent=self, title='Error', message='Bytes acquisition failed')
            return
        elif progress < 100: # Canceled
            return

        # Perform tests
        bytes_a, bytes_b = outputs

        test_bit_bias = self.var_bytes_bit_bias.get()
        if test_bit_bias:
            fig = tests_bytes.bytes_quick_test_bit_bias(bytes_a, bytes_b)
            fig.show()
            self.plots.append(fig)

        test_byte_bias = self.var_bytes_byte_bias.get()
        if test_byte_bias:
            fig = tests_bytes.bytes_quick_test_byte_bias(bytes_a, bytes_b)
            fig.show()
            self.plots.append(fig)

        test_correls = self.var_bytes_correl.get()
        if test_correls:
            fig = tests_bytes.bytes_quick_test_correls(bytes_a, bytes_b)
            fig.show()
            self.plots.append(fig)


    def nums_type_sel(self, tk_event=None):
        if self.cbb_nums_type.get() == 'INT':
            self.var_nums_range_min.set(0)
            self.var_nums_range_max.set(999)
            self.spb_nums_float_bins.state(['disabled'])
        else:
            self.var_nums_range_min.set(0)
            self.var_nums_range_max.set(1)
            self.spb_nums_float_bins.state(['!disabled'])


    def nums_test(self):
        # Read vars
        num_type = int if self.cbb_nums_type.get() == 'INT' else float
        n_nums = int(get_ammount_prefix_number(n=self.var_nums_n.get(), prefix=self.cbb_nums_n_prefix.get()))
        num_min = self.var_nums_range_min.get()
        num_max = self.var_nums_range_max.get()
        n_chunk = self.cfg.read('ACQUISITION', 'chunk_numbers', int)

        float_bins = self.var_nums_float_bins.get()
        if num_type is float and float_bins <= 0:
            tkm.showerror(parent=self, title='Error', message='Provide a Float bins > 0')
            return

        # Info
        self.lg.info('{}: Numbers - Bias Tests'.format(self.name))

        # Show win progress
        self.win_progress.show()

        # Acquire
        outputs, progress, time_str = self.rng_acq.get_numbers(rng=self.rng, n_nums=n_nums, n_chunk=n_chunk,
                                        num_type=num_type, num_min=num_min, num_max=num_max, out_file=False,
                                        threaded=False)

        # Hide win progress
        self.win_progress.hide()

        # Test outputs
        if progress == 0:
            tkm.showerror(parent=self, title='Error', message='Numbers acquisition failed')
            return
        elif progress < 100: # Canceled
            return

        # Perform tests
        nums = outputs[0]
        fig = tests_nums.nums_quick_test(nums, num_type, num_min, num_max, float_bins)
        fig.show()
        self.plots.append(fig)


    def pwm_freq_sel(self, tk_event=None):
        pwm_freqs = list(D_PWM_FREQ.keys())
        if self.cbb_pwm_freq.get() in [pwm_freqs[0], pwm_freqs[1]]:
            self.var_pwm_dc_min.set(4)
            self.var_pwm_dc_max.set(20)
            self.var_pwm_dc_step.set(1)
        else:
            self.var_pwm_dc_min.set(4)
            self.var_pwm_dc_max.set(40)
            self.var_pwm_dc_step.set(2)


    def pwm_test(self):
        # Read vars
        pwm_freq = self.cbb_pwm_freq.get()
        pwm_freq_id = D_PWM_FREQ[pwm_freq]

        pwm_dc_min = self.var_pwm_dc_min.get()
        pwm_dc_max = self.var_pwm_dc_max.get()
        pwm_dc_step = self.var_pwm_dc_step.get()
        pwm_dc_range = np.arange(pwm_dc_min, pwm_dc_max + 1, pwm_dc_step)
        pwm_dc_n = len(pwm_dc_range)
        if  pwm_dc_n == 0:
            tkm.showerror(parent=self, title='Error', message='Duty Cycle range')
            return

        n_pcs = int(get_ammount_prefix_number(n=self.var_pwm_pcs_n.get(), prefix=self.cbb_pwm_pcs_n_prefix.get()))
        if  pwm_dc_n >= 2**16:
                tkm.showerror(parent=self, title='Error', message='Provide PCs N < 65536')
                return

        si = self.var_pwm_si.get()

        pcs_a = np.zeros(shape=(pwm_dc_n, n_pcs), dtype=np.uint8)
        pcs_b = np.zeros(shape=(pwm_dc_n, n_pcs), dtype=np.uint8)

        # Info
        self.lg.info('{}: PWM - PC Evolution Test'.format(self.name))

        # Get RNG initial config
        si_init = self.rng.get_rng_setup()
        pwm_init = self.rng.get_pwm_setup()
        del pwm_init['freq_str']

        # Configure RNG Sampling interval
        self.rng.snd_rng_setup(sampling_interval_us=si)

        # Show progress window
        with self.win_progress:

            # Loop PCs acquisition
            canceled = False
            for i, dc in enumerate(pwm_dc_range):

                # Configure PWM
                self.rng.snd_pwm_setup(freq_id=pwm_freq_id, duty=dc)

                # Generate PCs
                pcs_a[i], pcs_b[i] = self.rng.get_rng_pulse_counts(n_counts=n_pcs, output_type='array', timeout=None)

                # Update progress
                self.win_progress.prog_update((i+1)/pwm_dc_n*100)

                # Canceled?
                if not self.win_progress.acquiring:
                    canceled = True
                    break

        # Restore RNG initial config
        self.rng.snd_rng_setup(**si_init)
        self.rng.snd_pwm_setup(**pwm_init)

        # Close progress window
        self.win_progress.hide()

        # Canceled?
        if canceled:
            return

        # Perform tests
        fig = tests_pwm.pwm_test(pcs_a, pcs_b, si, pwm_freq, pwm_dc_range)
        fig.show()
        self.plots.append(fig)


rava_subapp_quick_tests = {'class': RAVA_SUBAPP_QUICK_TESTS,
                           'menu_title': 'Quick Tests',
                           'show_button': True,
                           'use_rng': True
                           }