"""
Copyright (c) 2024 Gabriel Guerrer

Distributed under the MIT license - See LICENSE for details
"""

"""
The Detailed Tests sub-app analyzes the statistical distribution of bias and 
correlation metrics associated with the random bytes obtained through the 
Acquisition module. 

Moreover, it explores the normality of pulse count distributions across 
different sampling intervals. Additionally, it implements a visualization of 
the reports generated by the 
[NIST Tests](https://csrc.nist.rip/Projects/Random-Bit-Generation/Documentation-and-Software), 
offering an insightful representation of its results. 

Lastly, a reporting feature produces a PDF file summarizing the following 
analysis:
* Pulse Count distribution for 10K samples
* Bit and Byte bias tests for 125KB samples
* Bit bias, byte bias, and correlation distributions for 1000 x 125KB samples
* NIST Test results
The PDF document is designed to facilitate the quality assessment of a specific 
instance of the RAVA circuit.
"""

from mimetypes import init
import os.path

import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as tkm

import numpy as np
import matplotlib.pyplot as plt

from rng_rava.tk import RAVA_SUBAPP
from rng_rava.tk.acq import WIN_PROGRESS
from rng_rava.acq import get_ammount_prefix_number

import rng_rava_diag.tests_pcs as tests_pcs
import rng_rava_diag.tests_bytes as tests_bytes
import rng_rava_diag.tests_nums as tests_nums
import rng_rava_diag.tests_nist as tests_nist
import rng_rava_diag.tests_report as tests_report

### VARS

PAD = 10


### SUBAPP_TESTS

class RAVA_SUBAPP_DETAILED_TESTS(RAVA_SUBAPP):

    cfg_default_str = '''
    [DETAILED_TESTS]        
    file_pcs = 
    file_bytes_a = 
    file_bytes_b = 
    file_nums = 
    file_nist_a =
    file_nist_b =  
    path_report =
    '''

    def __init__(self, parent):
        # Initialize RAVA_SUBAPP
        name = 'RAVA_SUBAPP DETAILED TESTS'
        win_title = 'Detailed Tests'
        win_geometry = '510x280'
        win_resizable = False
        if not super().__init__(parent, name=name, win_title=win_title, win_geometry=win_geometry, win_resizable=win_resizable):
            return

        # WINDOWS
        self.win_progress = WIN_PROGRESS(self)
        self.win_progress.hide()

        # WIDGETS
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

        self.frm_nist = ttk.Frame(self, name='nist', padding=(0,PAD))
        self.frm_nist.grid(row=0, column=0, sticky='nsew')
        self.nist_widgets()

        self.frm_report = ttk.Frame(self, name='report', padding=(0,PAD))
        self.frm_report.grid(row=0, column=0, sticky='nsew')
        self.report_widgets()

        self.nb.add(self.frm_pcs, text=' Pulse Counts ')
        self.nb.add(self.frm_bytes, text=' Bytes ')
        self.nb.add(self.frm_nums, text=' Numbers ')
        self.nb.add(self.frm_nist, text=' NIST ')
        self.nb.add(self.frm_report, text=' Report ')

        # Key binds
        self.bind('<Control-Key-c>', lambda event=None: self.plots_close())

        ## Start
        self.plots = []

        # Config
        file_pcs = self.master.cfg_read('DETAILED_TESTS', 'file_pcs')
        file_bytes_a = self.master.cfg_read('DETAILED_TESTS', 'file_bytes_a')
        file_bytes_b = self.master.cfg_read('DETAILED_TESTS', 'file_bytes_b')
        file_nums = self.master.cfg_read('DETAILED_TESTS', 'file_nums')
        file_nist_a = self.master.cfg_read('DETAILED_TESTS', 'file_nist_a')
        file_nist_b = self.master.cfg_read('DETAILED_TESTS', 'file_nist_b')
        
        self.var_pcs_file.set(file_pcs)
        self.var_bytes_file_a.set(file_bytes_a)
        self.var_bytes_file_b.set(file_bytes_b)
        self.var_nums_file.set(file_nums)
        self.var_nist_file_a.set(file_nist_a)
        self.var_nist_file_b.set(file_nist_b)


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
        self.frm_pcs.columnconfigure([0], weight=1)
        self.frm_pcs.columnconfigure([1], weight=7)
        self.frm_pcs.rowconfigure([0,1,2,3], weight=1)

        # File
        self.lb_pcs_file = ttk.Label(self.frm_pcs, text='File (npz)')
        self.lb_pcs_file.grid(row=0, column=0)

        self.var_pcs_file = tk.StringVar(value='')
        self.en_pcs_file = ttk.Entry(self.frm_pcs, textvariable=self.var_pcs_file)
        self.en_pcs_file.grid(row=0, column=1, sticky='ew')

        self.bt_pcs_file_search = ttk.Button(self.frm_pcs, width=2, text='...', command=self.pcs_file_search)
        self.bt_pcs_file_search.grid(row=0, column=2, padx=PAD)

        # Parameters
        self.frm_pcs_pars = ttk.Frame(self.frm_pcs, name='pcs_pars')
        self.frm_pcs_pars.grid(row=1, column=0, columnspan=3, sticky='nsew')
        self.frm_pcs_pars.columnconfigure([0,1,2,3,4], weight=1)
        self.frm_pcs_pars.rowconfigure([0,1], weight=1)

        self.lb_pcs_n = ttk.Label(self.frm_pcs_pars, text='N PCs / test')
        self.lb_pcs_n.grid(row=0, column=0)

        self.var_pcs_n = tk.DoubleVar(value=10)
        self.spb_pcs_n = ttk.Spinbox(self.frm_pcs_pars, from_=1, to=999, increment=1, textvariable=self.var_pcs_n, width=8)
        self.spb_pcs_n.grid(row=0, column=1)

        self.cbb_pcs_n_prefix = ttk.Combobox(self.frm_pcs_pars, width=9)
        self.cbb_pcs_n_prefix.grid(row=0, column=2)
        self.cbb_pcs_n_prefix.state(['readonly'])
        self.cbb_pcs_n_prefix['values'] = ['', 'K', 'M', 'G', 'T']
        self.cbb_pcs_n_prefix.set('K')

        # Test
        self.bt_pcs_test_thbias = ttk.Button(self.frm_pcs_pars, text='Theoretical Bias', width=16, command=self.pcs_th_bias)
        self.bt_pcs_test_thbias.grid(row=1, column=0, columnspan=2, pady=PAD)

        self.bt_nist_plot = ttk.Button(self.frm_pcs_pars, text='Test', command=self.pcs_test)
        self.bt_nist_plot.grid(row=1, column=2, columnspan=2, pady=PAD)


    def bytes_widgets(self):
        self.frm_bytes.columnconfigure([0], weight=1)
        self.frm_bytes.columnconfigure([1], weight=7)
        self.frm_bytes.rowconfigure([0,1,2,3], weight=1)

        # Files
        self.lb_bytes_file_a = ttk.Label(self.frm_bytes, text='RNG A')
        self.lb_bytes_file_a.grid(row=0, column=0)

        self.var_bytes_file_a = tk.StringVar(value='')
        self.en_bytes_file_a = ttk.Entry(self.frm_bytes, textvariable=self.var_bytes_file_a)
        self.en_bytes_file_a.grid(row=0, column=1, sticky='ew')

        self.bt_bytes_file_a_search = ttk.Button(self.frm_bytes, width=2, text='...', command=self.bytes_file_a_search)
        self.bt_bytes_file_a_search.grid(row=0, column=2, padx=PAD)

        self.lb_bytes_file_b = ttk.Label(self.frm_bytes, text='RNG B')
        self.lb_bytes_file_b.grid(row=1, column=0)

        self.var_bytes_file_b = tk.StringVar(value='')
        self.en_bytes_file_b = ttk.Entry(self.frm_bytes, textvariable=self.var_bytes_file_b)
        self.en_bytes_file_b.grid(row=1, column=1, sticky='ew')

        self.bt_bytes_file_b_search = ttk.Button(self.frm_bytes, width=2, text='...', command=self.bytes_file_b_search)
        self.bt_bytes_file_b_search.grid(row=1, column=2, padx=PAD)

        # Parameters
        self.frm_bytes_pars = ttk.Frame(self.frm_bytes, name='bytes_pars')
        self.frm_bytes_pars.grid(row=2, column=0, columnspan=3, sticky='nsew')
        self.frm_bytes_pars.columnconfigure([0,1,2,3,4], weight=1)
        self.frm_bytes_pars.rowconfigure([0,1], weight=1)

        self.lb_bytes_n = ttk.Label(self.frm_bytes_pars, text='N Bytes / test')
        self.lb_bytes_n.grid(row=0, column=0)

        self.var_bytes_n = tk.DoubleVar(value=125)
        self.spb_bytes_n = ttk.Spinbox(self.frm_bytes_pars, from_=1, to=999, increment=1, textvariable=self.var_bytes_n, width=8)
        self.spb_bytes_n.grid(row=0, column=1)

        self.cbb_bytes_n_prefix = ttk.Combobox(self.frm_bytes_pars, width=9)
        self.cbb_bytes_n_prefix.grid(row=0, column=2)
        self.cbb_bytes_n_prefix.state(['readonly'])
        self.cbb_bytes_n_prefix['values'] = ['', 'K', 'M', 'G', 'T']
        self.cbb_bytes_n_prefix.set('K')

        self.lb_bytes_n_bins = ttk.Label(self.frm_bytes_pars, text='Fit bins')
        self.lb_bytes_n_bins.grid(row=1, column=0)

        self.var_bytes_n_bins = tk.IntVar(value=20)
        self.spb_bytes_n_bins = ttk.Spinbox(self.frm_bytes_pars, from_=1, to=999, increment=10, textvariable=self.var_bytes_n_bins, width=8)
        self.spb_bytes_n_bins.grid(row=1, column=1)

        # Test
        self.bt_bytes_plot = ttk.Button(self.frm_bytes, text='Test', command=self.bytes_test)
        self.bt_bytes_plot.grid(row=3, column=0, columnspan=4)


    def nums_widgets(self):
        self.frm_nums.columnconfigure([0], weight=1)
        self.frm_nums.columnconfigure([1], weight=7)
        self.frm_nums.rowconfigure([0,1,2,3], weight=1)

        # File
        self.lb_nums_file = ttk.Label(self.frm_nums, text='File (dat)')
        self.lb_nums_file.grid(row=0, column=0)

        self.var_nums_file = tk.StringVar(value='')
        self.en_nums_file = ttk.Entry(self.frm_nums, textvariable=self.var_nums_file)
        self.en_nums_file.grid(row=0, column=1, sticky='ew')

        self.bt_nums_file_search = ttk.Button(self.frm_nums, width=2, text='...', command=self.nums_file_search)
        self.bt_nums_file_search.grid(row=0, column=2, padx=PAD)

        # Parameters
        self.frm_nums_pars = ttk.Frame(self.frm_nums, name='nums_pars')
        self.frm_nums_pars.grid(row=1, column=0, columnspan=3, sticky='nsew')
        self.frm_nums_pars.columnconfigure([0,1,2,3,4], weight=1)
        self.frm_nums_pars.rowconfigure([0,1], weight=1)

        self.lb_nums_n = ttk.Label(self.frm_nums_pars, text='N Nums / test')
        self.lb_nums_n.grid(row=0, column=0)

        self.var_nums_n = tk.DoubleVar(value=10)
        self.spb_nums_n = ttk.Spinbox(self.frm_nums_pars, from_=1, to=999, increment=1, textvariable=self.var_nums_n, width=8)
        self.spb_nums_n.grid(row=0, column=1)

        self.cbb_nums_n_prefix = ttk.Combobox(self.frm_nums_pars, width=9)
        self.cbb_nums_n_prefix.grid(row=0, column=2)
        self.cbb_nums_n_prefix.state(['readonly'])
        self.cbb_nums_n_prefix['values'] = ['', 'K', 'M', 'G', 'T']
        self.cbb_nums_n_prefix.set('K')

        self.lb_float_n_bins = ttk.Label(self.frm_nums_pars, text='Fit bins (floats)')
        self.lb_float_n_bins.grid(row=1, column=0)

        self.var_float_n_bins = tk.IntVar(value=100)
        self.spb_float_n_bins = ttk.Spinbox(self.frm_nums_pars, from_=1, to=999, increment=10, textvariable=self.var_float_n_bins, width=8)
        self.spb_float_n_bins.grid(row=1, column=1)

        # Test
        self.bt_nums_plot = ttk.Button(self.frm_nums, text='Test', command=self.nums_test)
        self.bt_nums_plot.grid(row=2, column=0, columnspan=4, pady=PAD)


    def nist_widgets(self):
        self.frm_nist.columnconfigure([0], weight=1)
        self.frm_nist.columnconfigure([1], weight=7)
        self.frm_nist.rowconfigure([0,1,2,3,4], weight=1)

        # Files
        self.lb_nist_file_a = ttk.Label(self.frm_nist, text='Report A')
        self.lb_nist_file_a.grid(row=0, column=0)

        self.var_nist_file_a = tk.StringVar(value='')
        self.en_nist_file_a = ttk.Entry(self.frm_nist, textvariable=self.var_nist_file_a)
        self.en_nist_file_a.grid(row=0, column=1, sticky='ew')

        self.bt_nist_file_a_search = ttk.Button(self.frm_nist, width=2, text='...', command=self.nist_file_a_search)
        self.bt_nist_file_a_search.grid(row=0, column=2, padx=PAD)

        self.lb_nist_file_b = ttk.Label(self.frm_nist, text='Report B')
        self.lb_nist_file_b.grid(row=1, column=0)

        self.var_nist_file_b = tk.StringVar(value='')
        self.en_nist_file_b = ttk.Entry(self.frm_nist, textvariable=self.var_nist_file_b)
        self.en_nist_file_b.grid(row=1, column=1, sticky='ew')

        self.bt_nist_file_b_search = ttk.Button(self.frm_nist, width=2, text='...', command=self.nist_file_b_search)
        self.bt_nist_file_b_search.grid(row=1, column=2, padx=PAD)

        ## Tests
        self.bt_nist_plot = ttk.Button(self.frm_nist, text='Test', command=self.nist_test)
        self.bt_nist_plot.grid(row=2, column=0, columnspan=3)


    def report_widgets(self):
        self.frm_report.columnconfigure([0], weight=1)
        self.frm_report.columnconfigure([1], weight=7)
        self.frm_report.rowconfigure([0,1,2,3,4,5], weight=1)

        ## Files

        # PC
        self.lb_report_file_pc = ttk.Label(self.frm_report, text='PCs')
        self.lb_report_file_pc.grid(row=0, column=0)

        self.en_report_file_pc = ttk.Entry(self.frm_report, textvariable=self.var_pcs_file)
        self.en_report_file_pc.grid(row=0, column=1, sticky='ew')

        self.bt_report_file_pc_search = ttk.Button(self.frm_report, width=2, text='...', command=self.pcs_file_search)
        self.bt_report_file_pc_search.grid(row=0, column=2, padx=PAD)

        # Bytes
        self.lb_report_file_byte_a = ttk.Label(self.frm_report, text='Bytes A')
        self.lb_report_file_byte_a.grid(row=1, column=0)

        self.en_report_file_byte_a = ttk.Entry(self.frm_report, textvariable=self.var_bytes_file_a)
        self.en_report_file_byte_a.grid(row=1, column=1, sticky='ew')

        self.bt_report_file_byte_a_search = ttk.Button(self.frm_report, width=2, text='...', command=self.bytes_file_a_search)
        self.bt_report_file_byte_a_search.grid(row=1, column=2, padx=PAD)

        self.lb_report_file_byte_b = ttk.Label(self.frm_report, text='Bytes B')
        self.lb_report_file_byte_b.grid(row=2, column=0)

        self.en_report_file_byte_b = ttk.Entry(self.frm_report, textvariable=self.var_bytes_file_b)
        self.en_report_file_byte_b.grid(row=2, column=1, sticky='ew')

        self.bt_report_file_byte_b_search = ttk.Button(self.frm_report, width=2, text='...', command=self.bytes_file_b_search)
        self.bt_report_file_byte_b_search.grid(row=2, column=2, padx=PAD)

        # NIST
        self.lb_report_file_nist_a = ttk.Label(self.frm_report, text='NIST A')
        self.lb_report_file_nist_a.grid(row=3, column=0)

        self.var_report_file_nist_a = tk.StringVar(value='')
        self.en_report_file_nist_a = ttk.Entry(self.frm_report, textvariable=self.var_nist_file_a)
        self.en_report_file_nist_a.grid(row=3, column=1, sticky='ew')

        self.bt_report_file_nist_a_search = ttk.Button(self.frm_report, width=2, text='...', command=self.nist_file_a_search)
        self.bt_report_file_nist_a_search.grid(row=3, column=2, padx=PAD)

        self.lb_report_file_nist_b = ttk.Label(self.frm_report, text='NIST B')
        self.lb_report_file_nist_b.grid(row=4, column=0)

        self.var_report_file_nist_b = tk.StringVar(value='')
        self.en_report_file_nist_b = ttk.Entry(self.frm_report, textvariable=self.var_nist_file_b)
        self.en_report_file_nist_b.grid(row=4, column=1, sticky='ew')

        self.bt_report_file_nist_b_search = ttk.Button(self.frm_report, width=2, text='...', command=self.nist_file_b_search)
        self.bt_report_file_nist_b_search.grid(row=4, column=2, padx=PAD)

        ## Tests
        self.bt_report = ttk.Button(self.frm_report, text='Test', command=self.report_test)
        self.bt_report.grid(row=5, column=0, columnspan=3)


    def pcs_file_search(self):
        file_in0 = self.var_pcs_file.get()        
        file_dir_in0 = os.path.dirname(file_in0) if file_in0 else ''
        file_in = tk.filedialog.askopenfilename(parent=self, initialdir=file_dir_in0, 
                                                filetypes=[('Numpy Compressed File', '.npz')])
        if file_in:
            self.var_pcs_file.set(file_in)
            self.master.cfg_write('DETAILED_TESTS', 'file_pcs', file_in)


    def pcs_th_bias(self):
        # Info
        self.lg.info('{}: Pulse Count - Theretical Bias Test'.format(self.name))

        # Test
        fig = tests_pcs.pcs_thbias()
        fig.show()
        self.plots.append(fig)


    def pcs_test(self):
        # Get pars
        n_pcs_per_test = int(get_ammount_prefix_number(n=self.var_pcs_n.get(), prefix=self.cbb_pcs_n_prefix.get()))

        file_pcs = self.var_pcs_file.get()
        if not os.path.isfile(file_pcs):
            tkm.showerror(parent=self, message='Error', detail='File don\'t exists')
            return
        
        # Save cfg
        self.master.cfg_write('DETAILED_TESTS', 'file_pcs', file_pcs)

        # Get data
        file_dict = np.load(file_pcs)
        si_range = file_dict['si_range']
        pcs_a = file_dict['pcs_a']
        pcs_b = file_dict['pcs_b']

        # Process data
        pcs_3d_a = np.reshape(pcs_a, (pcs_a.shape[0], pcs_a.shape[1]//n_pcs_per_test, n_pcs_per_test))
        pcs_3d_b = np.reshape(pcs_b, (pcs_b.shape[0], pcs_b.shape[1]//n_pcs_per_test, n_pcs_per_test))

        # Run test and save plot
        figs = tests_pcs.pcs_detailed_test(si_range, pcs_3d_a, pcs_3d_b)
        for fig in figs:
            fig.show()
        self.plots.extend(figs)


    def bytes_file_a_search(self):
        file_in0 = self.var_bytes_file_a.get()        
        file_dir_in0 = os.path.dirname(file_in0) if file_in0 else ''
        file_in = tk.filedialog.askopenfilename(parent=self, initialdir=file_dir_in0, 
                                                filetypes=[('Binary File', '.bin')])
        if file_in:
            self.var_bytes_file_a.set(file_in)
            self.master.cfg_write('DETAILED_TESTS', 'file_bytes_a', file_in)


    def bytes_file_b_search(self):
        file_in0 = self.var_bytes_file_b.get()        
        file_dir_in0 = os.path.dirname(file_in0) if file_in0 else ''
        file_in = tk.filedialog.askopenfilename(parent=self, initialdir=file_dir_in0, 
                                                filetypes=[('Binary File', '.bin')])
        if file_in:
            self.var_bytes_file_b.set(file_in)
            self.master.cfg_write('DETAILED_TESTS', 'file_bytes_b', file_in)


    def bytes_test(self):
        # Get pars
        n_bytes_per_test = int(get_ammount_prefix_number(n=self.var_bytes_n.get(), prefix=self.cbb_bytes_n_prefix.get()))
        n_fit_bins = self.var_bytes_n_bins.get()

        file_bytes_a = self.var_bytes_file_a.get()
        file_bytes_b = self.var_bytes_file_b.get()

        if not os.path.isfile(file_bytes_a):
            tkm.showerror(parent=self, message='Error', detail='File A don\'t exists')
            return

        if not os.path.isfile(file_bytes_b):
            tkm.showerror(parent=self, message='Error', detail='File B don\'t exists')
            return
        
        # Save cfg
        self.master.cfg_write('DETAILED_TESTS', 'file_bytes_a', file_bytes_a)
        self.master.cfg_write('DETAILED_TESTS', 'file_bytes_b', file_bytes_b)

        # Get data
        with open(file_bytes_a, 'br') as f_a:
            bytes_a = np.frombuffer(f_a.read(), dtype=np.uint8)

        with open(file_bytes_b, 'br') as f_b:
            bytes_b = np.frombuffer(f_b.read(), dtype=np.uint8)

        # Process data
        n_tests = len(bytes_a) // n_bytes_per_test
        bytes_2d_a = np.reshape(bytes_a, (n_tests, n_bytes_per_test))
        bytes_2d_b = np.reshape(bytes_b, (n_tests, n_bytes_per_test))

        # Run test and save plot
        figs = tests_bytes.bytes_detailed_test(bytes_2d_a, bytes_2d_b, n_fit_bins)
        for fig in figs:
            fig.show()
        self.plots.extend(figs)


    def nums_file_search(self):
        file_in0 = self.var_nums_file.get()        
        file_dir_in0 = os.path.dirname(file_in0) if file_in0 else ''
        file_in = tk.filedialog.askopenfilename(parent=self, initialdir=file_dir_in0, 
                                                filetypes=[('Data File', '.dat'), ('Numpy File', '.npy')])
        if file_in:
            self.var_nums_file.set(file_in)
            self.master.cfg_write('DETAILED_TESTS', 'file_nums', file_in)


    def nums_test(self):
        # Get pars
        file_nums = self.var_nums_file.get()
        n_nums_per_test = int(get_ammount_prefix_number(n=self.var_nums_n.get(), prefix=self.cbb_nums_n_prefix.get()))
        float_n_bins = self.var_float_n_bins.get()

        if not os.path.isfile(file_nums):
            tkm.showerror(parent=self, message='Error', detail='File don\'t exists')
            return
        
        # Save cfg
        self.master.cfg_write('DETAILED_TESTS', 'file_nums', file_nums)

        ## Get data
        
        # Text file
        if '.dat' in file_nums:

            # Separator
            with open(file_nums, 'r') as f:
                txt = f.read(100)
                if ',' in txt:
                    sep = ','
                elif ';' in txt:
                    sep = ';'
                elif '\n' in txt:
                    sep = '\n'
                else:
                    tkm.showerror(parent=self, message='Error', detail='Can\'t detect the file separator')
                    return

            # Data type
            if 'INT' in file_nums:
                data_type = int
                float_n_bins = None
            elif 'FLOAT' in file_nums:
                data_type = float
            else:
                tkm.showerror(parent=self, message='Error', detail='Can\'t detect the file\'s data type')
                return

            nums = np.fromfile(file_nums, dtype=data_type, sep=sep)

        # Binary File
        elif '.npy' in file_nums:
            nums = np.load(file_nums)

        ## Process data
        n_tests = len(nums) // n_nums_per_test
        nums_2d = np.reshape(nums, (n_tests, n_nums_per_test))

        ## Run test and save plot
        fig = tests_nums.nums_detailed_test(nums_2d, float_n_bins)
        fig.show()
        self.plots.append(fig)


    def nist_file_a_search(self):
        file_in0 = self.var_nist_file_a.get()        
        file_dir_in0 = os.path.dirname(file_in0) if file_in0 else ''
        file_in = tk.filedialog.askopenfilename(parent=self, initialdir=file_dir_in0, 
                                                filetypes=[('NIST Report File', '.txt')])
        if file_in:
            self.var_nist_file_a.set(file_in)
            self.master.cfg_write('DETAILED_TESTS', 'file_nist_a', file_in)


    def nist_file_b_search(self):
        file_in0 = self.var_nist_file_b.get()        
        file_dir_in0 = os.path.dirname(file_in0) if file_in0 else ''
        file_in = tk.filedialog.askopenfilename(parent=self, initialdir=file_dir_in0, 
                                                filetypes=[('NIST Report File', '.txt')])
        if file_in:
            self.var_nist_file_b.set(file_in)
            self.master.cfg_write('DETAILED_TESTS', 'file_nist_b', file_in)


    def nist_test(self):
        # Get pars
        file_nist_a = self.var_nist_file_a.get()
        file_nist_b = self.var_nist_file_b.get()

        if not os.path.isfile(file_nist_a):
            tkm.showerror(parent=self, message='Error', detail='File A don\'t exists')
            return

        if not os.path.isfile(file_nist_b):
            tkm.showerror(parent=self, message='Error', detail='File B don\'t exists')
            return
        
        # Save cfg
        self.master.cfg_write('DETAILED_TESTS', 'file_nist_a', file_nist_a)
        self.master.cfg_write('DETAILED_TESTS', 'file_nist_b', file_nist_b)

        # Run test and save plot
        fig = tests_nist.nist_test(file_nist_a, file_nist_b)
        fig.show()
        self.plots.append(fig)


    def report_test(self):
        # Get pars
        file_pcs = self.var_pcs_file.get()
        file_bytes_a = self.var_bytes_file_a.get()
        file_bytes_b = self.var_bytes_file_b.get()        
        file_nist_a = self.var_nist_file_a.get()
        file_nist_b = self.var_nist_file_b.get()        

        # Check file exists; NIST reports aren't mandatory
        if not os.path.isfile(file_pcs):
            tkm.showerror(parent=self, message='Error', detail='Pulse Counts file doesn\'t exists')
            return
        
        if not os.path.isfile(file_bytes_a):
            tkm.showerror(parent=self, message='Error', detail='Bytes A file doesn\'t exists')
            return

        if not os.path.isfile(file_bytes_b):
            tkm.showerror(parent=self, message='Error', detail='Bytes B file doesn\'t exists')
            return        
       
        # Output file
        path_report0 = self.master.cfg_read('DETAILED_TESTS', 'path_report')
        path_report = tk.filedialog.askdirectory(parent=self, initialdir=path_report0, mustexist=True)
        if path_report:
            self.master.cfg_write('DETAILED_TESTS', 'path_report', path_report)

            # Generate Report
            tests_report.report(self, path_report, file_pcs, file_bytes_a, file_bytes_b, file_nist_a, file_nist_b)
        return 
        

rava_subapp_detailed_tests = {'class': RAVA_SUBAPP_DETAILED_TESTS,
                            'menu_title': 'Detailed Tests',
                            'show_button': True,
                            'use_rng': False
                            }