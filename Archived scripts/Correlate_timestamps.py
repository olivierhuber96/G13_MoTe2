# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:06:04 2023

@author: SSPDs
"""

from RemoteCorrelationAnalysis import correlate_parallel, correlate_timestamps
import os
import time
import pickle
from datetime import datetime
fmt = '%Y.%m.%d %H:%M'
#%%

fpath = r'Z:\Labs\G8\dipolaritonsInFiberCavity\20231109\data'
plot_path = r'Z:\Labs\G8\dipolaritonsInFiberCavity\20231109\plots'


print(datetime.now().strftime(fmt) + ': Waiting for 2h to start correlating')
# time.sleep(2*60*60)
fname = r'CorrelationFPGA_23-11-09--10-10-41_.p'
print(datetime.now().strftime(fmt) + ': Started correlating:\n'+fname)
with open(fpath+os.sep+fname, 'rb') as file:
    exp = pickle.load(file)
    
argument = {'data_path': fpath, 'plot_path': plot_path,
            'post_selection_bins': exp['post_selection_bins'],
            'tau_binning': exp['tau_binning'], 'laser_sweep': True,
            'Sweep_npoints': exp['Sweep_npoints'], 'Sweep_dt': exp['Sweep_dt']}

for i in range(len(exp['T2_files'])):
    argument['input_file'] = exp['T2_files'][i]
    correlate_parallel(argument)
    time.sleep(2)


# time.sleep(200*60)

#%%
fpath = r'Z:\Labs\G8\dipolaritonsInFiberCavity\20231109\data'
plot_path = r'Z:\Labs\G8\dipolaritonsInFiberCavity\20231109\plots'
fname = r'CorrelationFPGA_23-11-09--13-34-31_.p'
print(datetime.now().strftime(fmt) + ': Started correlating:\n'+fname)
with open(fpath+os.sep+fname, 'rb') as file:
    exp = pickle.load(file)
    
argument = {'data_path': fpath, 'plot_path': plot_path,
            'post_selection_bins': exp['post_selection_bins'],
            'tau_binning': exp['tau_binning'], 'laser_sweep': True,
            'Sweep_npoints': exp['Sweep_npoints'], 'Sweep_dt': exp['Sweep_dt']}

for i in range(len(exp['T2_files'])):
    argument['input_file'] = exp['T2_files'][i]
    correlate_parallel(argument)
    time.sleep(2)

print(datetime.now().strftime(fmt) + ': Waiting for 152min to start correlating')
# time.sleep(152*60)

#%%
fpath = r'Z:\Labs\G8\dipolaritonsInFiberCavity\20231109\data'
plot_path = r'Z:\Labs\G8\dipolaritonsInFiberCavity\20231109\plots'
fname = r'CorrelationFPGA_23-11-09--17-15-29_.p'
print(datetime.now().strftime(fmt) + ': Started correlating:\n'+fname)
with open(fpath+os.sep+fname, 'rb') as file:
    exp = pickle.load(file)
    
argument = {'data_path': fpath, 'plot_path': plot_path,
            'post_selection_bins': exp['post_selection_bins'],
            'tau_binning': exp['tau_binning'], 'laser_sweep': True,
            'Sweep_npoints': exp['Sweep_npoints'], 'Sweep_dt': exp['Sweep_dt']}

for i in range(len(exp['T2_files'])):
    argument['input_file'] = exp['T2_files'][i]
    correlate_parallel(argument)
    time.sleep(2)
    
#%%    
fpath = r'Z:\Labs\G8\dipolaritonsInFiberCavity\20231110\data'
plot_path = r'Z:\Labs\G8\dipolaritonsInFiberCavity\20231110\plots'  
fname = r'CorrelationFPGA_23-11-10--10-17-26_.p'
print(datetime.now().strftime(fmt) + ': Started correlating:\n'+fname)
with open(fpath+os.sep+fname, 'rb') as file:
    exp = pickle.load(file)
    
argument = {'data_path': fpath, 'plot_path': plot_path,
            'post_selection_bins': exp['post_selection_bins'],
            'tau_binning': exp['tau_binning'], 'laser_sweep': True,
            'Sweep_npoints': exp['Sweep_npoints'], 'Sweep_dt': exp['Sweep_dt']}

for i in range(len(exp['T2_files'])):
    argument['input_file'] = exp['T2_files'][i]
    correlate_parallel(argument)
    time.sleep(2)  

#%%
fpath = r'Q:\Correlation_0D\data\231218'
plot_path = r'Q:\Correlation_0D\plots\231218'  
fname = r'231218_results_14-00-48_vtg1.55_vdiff21_wm758.b'
print(datetime.now().strftime(fmt) + ': Started correlating:\n'+fname)
# with open(fpath+os.sep+fname, 'rb') as file:
#     exp = pickle.load(file)

import numpy as np
argument = {'data_path': fpath, 'plot_path': plot_path,
            'post_selection_bins': np.linspace(0, 3000, 21)*1e3,
            'tau_binning': 2, 
            'laser_sweep': False,
            'tau_max' : 512*1000,
            # 'chunk_bins' : np.linspace(0, 8e6, 8),
            }
argument['input_file'] = "Q:\\Correlation_0D\\" + fname

print(correlate_timestamps(argument))
