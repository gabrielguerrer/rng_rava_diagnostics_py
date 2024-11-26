"""
Copyright (c) 2024 Gabriel Guerrer

Distributed under the MIT license - See LICENSE for details
"""

"""
This code evokes the RAVA's Diagnostics Tk application.
"""

import rng_rava as rava
import rng_rava_diag as rava_diag

TITLE = 'RAVA Diagnostics'

rava_subapp_control_panel = rava.tk.ctrlp.rava_subapp_control_panel
rava_subapp_control_panel['show_button'] = False
SUBAPPS = [rava_subapp_control_panel,
           rava_diag.rava_subapp_quick_tests,
           rava.tk.acq.rava_subapp_acquisition,
           rava_diag.rava_subapp_detailed_tests]


def main():    
    # RAVA main app
    tkapp = rava.tk.RAVA_APP(title=TITLE, subapp_dicts=SUBAPPS, rava_class=rava.RAVA_RNG, cfg_log_name='rava_diag')

    # Enter Tk loop
    tkapp.mainloop()


if __name__ == '__main__':
    main()