# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:58:29 2023

@author: SSPDs
"""
import safeInterrupt
import numpy as np
import time
from livePlotting_livio import LivePlot


import Hydraharp
hydra = Hydraharp.hydraharp()
hydra.opendefaults()
hydra.SetInputCFD(1, 390, 20)
hydra.SetSyncCFD(390, 20)
hydra.SetSyncDiv(1)
hydra.SetSyncChannelOffset(0)





lp = LivePlot()

def test_hydra(lp):
    safeInterrupt.stop_flag = False

    #acq_time = 0.5
    refr_time = 0.05
    totPoints = 200
    measure = True

    lp.win.clear()
    plt1 = lp.win.addPlot(row=1, col=1)
    plt1.setLabel('left', 'counts', 'Hz')
    plt1.setLabel('bottom', 'time', 's')
    line1 = plt1.plot(x=[0], y=[0], symbol='o', symbolPen='k')

    nr_points = 0
    y_vals = []

    while measure:
        if safeInterrupt.stop_flag is True:
            measure = False
        res = hydra.GetSyncRate()+hydra.GetCountRate(channel=1)
        
        nr_points += 1
        if nr_points < totPoints:
            y_vals.append(res)
            x_vals = np.arange(nr_points) - nr_points +1
            x_vals = x_vals*refr_time
        else:
            y_vals.append(res)
            del y_vals[0]
        line1.setData(x=x_vals, y=y_vals)
        time.sleep(refr_time)
        

print('StartPlotting')
try:
    test_hydra(lp)
except:
    hydra.close()
print('Finished Plotting, closing hydra')        

hydra.close()

