# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 21:34:45 2017

@author: Thomas Fink, Quantum  Photonics Group, ETH Zurich

This script initializes a temperature sensor with the indicated serialnumber,
reads out the temperature every 'interval' seconds, and saves the data to files
with the 'filename' prefix on the group drive.

The json file is the one that is displayed on the lab website and only includes
the last 576 values (i.e. two days for 5min readout intervals), whereas the txt
file contains all the data (accessible via the 'Download' button on the lab
website).

"""

import time
import datetime
import logging
import sys, os
from TemperatureMonitor import TSP
import json


#%% Input by user; should be adapter for each lab / sensor

serialnumber    = 'M00276595'   # sensor serial number
filename        = 'g13'          # filename prefix
interval        = 300           # readout interval (seconds)


#%% Initialize logger

logging.basicConfig(stream = sys.stdout,
                    level = logging.WARNING,
                    format = '%(asctime)s - %(name)s - %(levelname)s - '
                             '%(message)s',
                    datefmt = '%Y-%m-%d, %H:%M:%S',
                    )
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)


#%% Adjust datetime difference; on some systems datetime.now() returns UTC time

tzdiff = datetime.datetime.now()-datetime.datetime.utcnow()


#%% Initialize temperature sensor

tsp = TSP(serialnumber)
root_logger.info("TSP connected.")


#%% Initialize data dictionary

data            = {}
#data['time']    = [(datetime.datetime.now()+tzdiff).strftime("%Y-%m-%d %H:%M:%S")]*576
data['time']    = [(datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")]*576
data['temp']    = [tsp.temp[1]]*576


#%% Start infinite loop; adapt filename prefix below

while True:
#    data['time'].append((datetime.datetime.now()+tzdiff).strftime("%Y-%m-%d %H:%M:%S"))
    data['time'].append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    data['time'].pop(0)
    data['temp'].append(tsp.temp[1])
    data['temp'].pop(0)
    folder = r'\\windata.phys.ethz.ch\qpgroup\public_html\temperature'
    try:
        with open(folder + os.sep + 'temp_' + filename + '.json',
                  'w',
                  ) as file:
            json.dump(data,file)
        with open(folder
                  + os.sep
                  + 'log_'
                  + filename
                  + '.txt',
                  'a',
                  ) as f:
            f.write('%s, %s\n' % (data['time'][-1], data['temp'][-1]))
    except:
        root_logger.error("Could not save data file.")
#    time.sleep(10)
    time.sleep(interval)


#%% Close temperature sensor

tsp.close()
root_logger.info("TSP disconnected.")