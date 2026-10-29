# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:06:37 2023

@author: SSPDs
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
#%%
exp = {}
exp['CFD'] = [420, 390]
import Hydraharp
hydra = Hydraharp.hydraharp()
hydra.opendefaults()
hydra.SetInputCFD(1, exp['CFD'][1], 20)
hydra.SetSyncCFD(exp['CFD'][0], 20)
hydra.SetSyncDiv(1)
hydra.SetSyncChannelOffset(0)


#%%

exp['tint'] =300
hydra.ClearHistMem()
hydra.StartMeas(tacq=exp['tint']*1000)
done = False

while not done:
    if hydra.CTCStatus() == 1:
        done = True

hist = hydra.GetHistogram(channel=1)
hydra.StopMeas()
#%%
Npoints = len(hist)
dt = 2
t = np.linspace(0,2*Npoints,Npoints)
binsize = 10
bins = np.arange(np.min(hist), np.max(hist), binsize)
dist, _ = np.histogram(hist, bins)

fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.plot(t*1e-3, hist)
ax.set_xlabel('time (ns)')
ax.set_ylabel('coincidences')
ax.set_title('CW histogram (nf)')

mean = np.mean(hist)
std = np.std(hist)
ctax = np.linspace(np.min(hist), np.max(hist), 200)
gauss = Npoints*binsize/std/np.sqrt(2*np.pi)*np.exp(-0.5*(ctax-mean)**2/std**2)
ax = fig.add_subplot(2,1,2)
ax.bar(bins[1:]-5, dist, width=0.8*binsize)
ax.plot(ctax, gauss,color='C1')
ax.set_xlabel('coincidences')
ax.set_ylabel('occurances')
text = r'$\mu$ = ' + str(np.round(mean)) + 'Cts \n'
text += r'$\sigma$ = ' + str(np.round(std)) + 'Cts'
ax.text(800,6000,text)

fpath = r'Z:\Labs\G13\SSPDs\230315'
pltname = 'cw_correlations.png'
plt.savefig(fpath+os.sep+pltname, dpi=300)

#%%
exp['hist'] = hist
exp['t'] = t

with open(fpath+'cw_correlations.p', 'wb') as file:
    pickle.dump(exp, file)
#%%
hydra.close()