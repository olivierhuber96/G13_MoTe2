# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:30:25 2023

@author: SSPDs
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
fpath = r'Z:\Labs\G13\SSPDs\230316'
#%%
exp = {}
exp['CFD'] = [390, 390]
import Hydraharp
hydra = Hydraharp.hydraharp()
hydra.opendefaults()
hydra.SetInputCFD(1, exp['CFD'][1], 20)
hydra.SetSyncCFD(exp['CFD'][0], 20)
hydra.SetSyncDiv(1)
hydra.SetSyncChannelOffset(0)

#%%
def Timetrace(Npoints, intervall=0.1):
    fname = 'Timetrace_' + time.strftime("%y-%m-%d--%H-%M-%S", time.localtime())
    t0 = time.time()
    t = np.zeros(Npoints)
    ctrate = np.zeros((2,Npoints))
    
    for i in range(Npoints):
        t1 = time.time()-t0
        ctrate[0,i] = hydra.GetSyncRate()
        ctrate[1,i] = hydra.GetCountRate(channel=1)
        t2 = time.time() - t0
        time.sleep(intervall)
        t[i] = np.mean([t1,t2])
    return t, ctrate, fname


#%%
t, ctrate, fname = Timetrace(500, intervall=0.1)
#%%

plt.figure()
plt.plot(t, ctrate.T)
plt.xlabel('time s')
plt.ylabel('Counts Hz')
plt.title(fname+'RedRubber')
plt.tight_layout()
#%%
fft0 = np.fft.fft(ctrate[0])
fft1 = np.fft.fft(ctrate[1])
f = np.fft.fftfreq(len(t), t[1]-t[0])
plt.figure()
plt.loglog(f, abs(fft0.real)**2)
plt.loglog(f, abs(fft1.real)**2)
plt.xlim(0.5e-1,5)
plt.xlabel('f Hz')
plt.ylabel('PSD')
#%%
plt.savefig(fpath+os.sep+fname+'.png', dpi=300, transparent=False)
np.savetxt(fpath+os.sep+fname+'.txt', [t, ctrate[0], ctrate[1]], header='t\tSync\tChannel1')
#%%
hydra.close()
