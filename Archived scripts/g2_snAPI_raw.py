# -*- coding: utf-8 -*-
"""
PhotonCorrelation -- uses snAPI.Main.Raw to acquire TTTR (T2) records and compute photon correlations.
Notes:
    - This implementation focuses on MeasMode.T2 (timetag records). 
    - Docs: Raw.decode functions: isSpecial(), isMarker(), channel(), timeTag_T2(), etc.
Author: Oli
Created: 2025-11-04
"""
import os
import time
from datetime import datetime
import threading
import numpy as np
import Pyro4
import matplotlib as mpl
mpl.use("Agg")   # non-interactive backend to be able to save plots in a thread that is not the main thread
import matplotlib
matplotlib.use('TkAgg', force=True)
from matplotlib import pyplot as plt
from time import gmtime, strftime
import Pyro4
from snAPI.Main import snAPI, MeasMode
from typing import Tuple, Optional, List

# Pyro configuration
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
Pyro4.config.SERVERTYPE = "thread"
Pyro4.config.COMMTIMEOUT = 0

def curtime():
    """Return UTC timestamp string."""
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())

@Pyro4.expose
class PhotonCorrelation(object):
    def __init__(self, config_path="hh_config/HH.ini", verbose=True,
                 save_dir=r"C:\Users\SSPDs\Documents\Offline SSPD Data"):
        self.verbose = verbose
        self.config_path = config_path
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.sn = None
        self.raw = None
        self.deviceConfig = None
        self.data = None
        self._resolution_sec = None

        if self.verbose:
            print(curtime(), f"PhotonCorrelation object created. Save dir: {self.save_dir}")

    # ------------------------------------------------------------------
    def init_device(self, save_dir=None):
        """Initialize snAPI device in T2 mode and optionally set new save directory."""
        if save_dir:
            self.save_dir = os.path.abspath(save_dir)
            os.makedirs(self.save_dir, exist_ok=True)
        
        self.sn = snAPI()
        self.raw = self.sn.raw
        print(curtime(), "Detecting and connecting to device...")
        self.sn.getDeviceIDs()
        self.sn.getDevice()
        self.sn.initDevice(MeasMode.T2)

        # Load config if exists
        if os.path.isfile(self.config_path):
            self.sn.loadIniConfig(self.config_path)
            print(curtime(), f"Loaded INI config: {self.config_path}")
        else:
            print(curtime(), f"WARNING: INI config not found at {self.config_path}")

        # Set save directory for PTU files
        self.sn.deviceConfig['SaveDir'] = self.save_dir

        print(curtime(), f"Save directory set to: {self.save_dir}")

        # Resolution in seconds
        self.deviceConfig = self.sn.deviceConfig
        self._resolution_sec = float(self.deviceConfig.get("Resolution", 1.0)) * 1e-12

        print(curtime(), "HydraHarp device initialized in T2 mode.")

    # ------------------------------------------------------------------
    def _generate_filename(self, prefix="measurement"):
        """Generate timestamped PTU filename on C-drive."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.ptu"
        full_path = os.path.join(self.save_dir, filename)
        return full_path

    def start_measure_t2(self, acq_time_ms: int = 1000,
                         size: int = 134217728,
                         wait_finished: bool = True,
                         save_ptu: bool = True):
        """
        acq_time: in milliseconds
        size    : file size must be large enough otherwise data will be overwritten
        save_ptu: allows to save data on C-drive
        """
        if not self.sn:
            raise RuntimeError("Device not initialized. Call init_device() first.")

        # Ensure save directory is active
        self.sn.deviceConfig['SaveDir'] = self.save_dir
        filepath = self._generate_filename()

        print(curtime(), f"Starting T2 measurement: {filepath}")

        ok = self.raw.measure(acqTime=acq_time_ms, size=size,
                              waitFinished=wait_finished, savePTU=save_ptu)
        if not ok:
            raise RuntimeError("sn.raw.measure() failed")
        
        print(curtime(), "Measurement complete.")


if __name__ == "__main__":
    # create snAPI object and prepare device (this is example code; adapt to your device)
    svc = PhotonCorrelation()
    daemon = Pyro4.Daemon(host='g13-sspd.dhcp-int.phys.ethz.ch', port=9096)
    uri = daemon.register(svc, objectId="G2_snAPI_raw")
    print("snAPI URI:", uri)
    daemon.requestLoop()

    