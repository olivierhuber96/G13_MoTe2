import safeInterrupt
import daq_counter
import numpy as np

from livePlotting_livio import LivePlot

lp = LivePlot()

def test_apd(lp):
    safeInterrupt.stop_flag = False

    acq_time = 0.5
    sample_rate = 1e3

    measure = True
    vals = []

    lp.win.clear()
    plt1 = lp.win.addPlot(row=1, col=1)
    plt1.setLabel('left', 'counts', 'Hz')
    line1 = plt1.plot(x=[0], y=[0], symbol='o', symbolPen='k')

    nr_points = 0
    y_vals = []

    while measure:
        if safeInterrupt.stop_flag is True:
            measure = False
        
        res = daq_counter.get_one_ctrs(sample_rate, acq_time)
        nr_points += 1
        y_vals.append(res[-1] / acq_time)
        line1.setData(x=np.arange(nr_points), y=y_vals)