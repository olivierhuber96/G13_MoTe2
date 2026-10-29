# -*- coding: utf-8 -*-
"""
Extended class for PicoQuant snAPI G(2) T2 measurements.
Now includes:
  - Save to CSV
  - Optional threaded live plotting
Author: Oli
Created: 2025-11-04
"""

import os
import time
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

# Pyro configuration
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
Pyro4.config.SERVERTYPE = "thread"
Pyro4.config.COMMTIMEOUT = 0

def curtime():
    """Return UTC timestamp string."""
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())

@Pyro4.expose
class G2_snAPI(object):
    """Reusable class for PicoQuant snAPI g(2) measurements in T2 mode."""
    def __init__(self, config_path="hh_config/HH.ini", verbose=True):
        self.verbose = verbose
        self.config_path = config_path
        self.sn = None
        self.data = None
        self.bins = None
        self.fig = None
        self.ax = None
        self.line = None
        self._plot_thread = None
        self._stop_plot = threading.Event()
        if self.verbose:
            print(curtime(), "G2Measurement object created.")

    # ------------------------------------------------------------------
    def init_device(self):
        """Initialize snAPI device in T2 mode."""
        self.sn = snAPI()
        print(curtime(), "Detecting and connecting to device...")
        self.sn.getDeviceIDs()
        self.sn.getDevice()
        self.sn.initDevice(MeasMode.T2)
        
        if os.path.isfile(self.config_path):
            self.sn.loadIniConfig(self.config_path)
            print(curtime(), f"Loaded INI config: {self.config_path}")
        else:
            print(curtime(), f"WARNING: INI config not found at {self.config_path}")

    # ------------------------------------------------------------------
    def set_offsets(self, channel=1, offset=0):
        """Set input channel offset."""
        self._check_device()
        self.sn.device.setInputChannelOffset(channel, offset)
        if self.verbose:
            print(curtime(), f"Set input offset for channel {channel} → {offset}")

    # ------------------------------------------------------------------
    def set_g2_parameters(self, ch1=1, ch2=2, window_size=200e6, time_bin=40e3, normalize=False):
        """Configure correlation parameters."""
        self._check_device()
        self.sn.correlation.setG2Parameters(ch1, ch2, window_size, time_bin, normalize)
        print(curtime(), f"Configured g2 parameters: ({ch1}, {ch2}, {window_size}, {time_bin}, {normalize})")

    # ------------------------------------------------------------------
    def start_measurement(self,acq_time=int(5*60e3),save_ptu=False,live_plot=True,threaded_plot=False,
                               poll_interval=2,autosave=False,save_path=None):
        """Start the g(2) measurement.
        acq_time in milliseconds"""
        self._check_device()
        print(curtime(), "Starting correlation measurement...")
        self.sn.correlation.measure(acq_time, waitFinished=False, savePTU=save_ptu)
        if live_plot:
            self._setup_plot()

        # If threaded plotting is requested
        if threaded_plot and live_plot:
            self._plot_thread = threading.Thread(target=self._plot_loop, args=(poll_interval,))
            self._plot_thread.start()
            return  
        else:
            while True:
                finished = self.sn.correlation.isFinished()
                data, bins = self.sn.correlation.getG2Data()
                if data is not None and bins is not None:
                    self.data = np.asarray(data)
                    self.bins = np.asarray(bins)
                    if live_plot:
                        self._update_plot()
                if finished:
                    break
                time.sleep(poll_interval)

        print(curtime(), "Measurement completed.")

        if autosave:
            self.save_data(save_path)

        if live_plot:
            plt.ioff()
            plt.show(block=True)

    # ------------------------------------------------------------------
    def _plot_loop(self, poll_interval):
        """Run live plotting in a separate thread."""
        print(curtime(), "Live plotting thread started.")
        while not self._stop_plot.is_set():
            finished = self.sn.correlation.isFinished()
            data, bins = self.sn.correlation.getG2Data()
            if data is not None and bins is not None:
                self.data = np.asarray(data)
                self.bins = np.asarray(bins)
                self._update_plot()
            if finished:
                break
            time.sleep(poll_interval)
        self._stop_plot.set()
        print(curtime(), "Live plotting thread finished.")

    # ------------------------------------------------------------------
    def _setup_plot(self):
        """Prepare the live matplotlib figure."""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.line, = self.ax.plot([], [], linewidth=2.0, label='g(2)')
        self.ax.set_xlabel('Time [s]')
        self.ax.set_ylabel('g(2)')
        self.ax.set_title("g(2) correlation")
        self.ax.legend()
        plt.show(block=False)

    def _update_plot(self):
        """Update live plot with latest data."""
        if self.line is None or self.bins is None or self.data is None:
            return
        self.line.set_xdata(self.bins)
        self.line.set_ydata(self.data)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.1)

    # ------------------------------------------------------------------
    def save_data(self, filepath=None, fmt="csv"):
        """Save g(2) data to CSV."""
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            
        if self.data is None or self.bins is None:
            raise RuntimeError("No data to save — run a measurement first.")

        timestamp = strftime("%Y%m%d_%H%M%S", gmtime())
        if filepath is None:
            filepath = f"g2_{timestamp}.{fmt}"

        if fmt == "csv":
            np.savetxt(filepath, np.column_stack((self.bins, self.data)),
                       delimiter=",", header="Time(s),g2", comments="")
            print(curtime(), f"Saved data at {filepath}")
        else:
            raise ValueError("Unsupported format. Use 'csv'.")

    # ------------------------------------------------------------------
    def get_data(self):
        """Return (bins, g2) arrays."""
        if self.data is None or self.bins is None:
            raise RuntimeError("No data available yet.")
        return self.bins, self.data

    # ------------------------------------------------------------------
    def close(self):
        """Close snAPI device and stop threads."""
        if self._plot_thread and self._plot_thread.is_alive():
            self._stop_plot.set()
            self._plot_thread.join(timeout=2)

        if self.sn is not None:
            try:
                if hasattr(self.sn, "close"):
                    self.sn.close()
                elif hasattr(self.sn, "shutdown"):
                    self.sn.shutdown()
                elif hasattr(self.sn, "device") and hasattr(self.sn.device, "close"):
                    self.sn.device.close()
            except Exception as e:
                print(curtime(), "Warning during close:", e)
        print(curtime(), "Device connection closed.")

    # ------------------------------------------------------------------
    def _check_device(self):
        """Ensure device is initialized."""
        if self.sn is None:
            raise RuntimeError("Device not initialized. Call init_device() first.")


if __name__ == "__main__":
    svc = G2_snAPI()
    daemon = Pyro4.Daemon(host='g13-sspd.dhcp-int.phys.ethz.ch', port=9096)
    uri = daemon.register(svc, objectId="G2_snAPI")
    print("snAPI URI:", uri)
    daemon.requestLoop()
    #g2 = G2Measurement(config_path="hh_config/HH.ini", verbose=True)
    #g2.init_device()
    #g2.set_offsets(1, 0)
    #g2.set_g2_parameters(1, 2, windowSize=200e6, binWidth=1e3, normalization=False) #windowSize, binWidth in ps

    # Run measurement with threaded plotting and automatic save
    #g2.start_measurement(acq_time=5*60e3,save_ptu=False,live_plot=True,
    #                     threaded_plot=True,autosave=True,save_path="test_g2.csv")
    #bins, data = g2.get_data()
    #plt.figure()
    #plt.histogram(bins,data)
    #g2.close()
