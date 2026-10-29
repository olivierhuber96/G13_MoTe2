
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 10:48:27 2016

@author: Thomas Fink, Quantum  Photonics Group, ETH Zurich

This application allows to monitor the readout of different lab devices. It is
based on PyQt / PyQtGraph and incorporates a thread getting new data for each
device and emitting a signal that is caught by the corresponding plot window.
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree
import numpy as np
from matplotlib import cm
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.console import ConsoleWidget
import collections
import sys
import Pyro4
import logging
import nidaqmx 

from Picoharp import picoharp
from Hydraharp import hydraharp
from nidaqmx.constants import AcquisitionType
#from nidaqmx import AnalogInputTask, CounterInputTask, CounterOutputTask
#import PyCapture2 as pc
#from ThorlabsPM100 import ThorlabsPM100

class LivePlotter(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(LivePlotter, self).__init__()
        
        self._GenWindow()
        self._GenDocks()
        self._InitParams()
        self._GenActions()
        self._GenThreads()
        self._GenStatusbar()
        self._GenMenuBar()
        self._GenToolBar()        
        self._GenData()
        self._GenPlotWidgets()
        self._SetRangePlots()
        self._DockWidgets()
        self.show()
        self.showMaximized()
        logging.basicConfig(stream = sys.stdout,
                            level  = logging.INFO,
                            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d, %H:%M:%S',
                            )
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.INFO)
        
    def close(self):
        [thread.stop() for key, thread in self.threads.__dict__.items() 
            if not key.startswith('__') if not thread == None
        ]
        sys.exit()

    def _showabout(self):
        txt = '''This application has been built by
                 Thomas Fink
                 Quantum Photonics Group
                 ETH Zurich
                 
                 December 2016'''
        self.widgets.about = QtWidgets.QDialog()
        self.widgets.about.setWindowTitle('About')
        self.widgets.about.setWindowModality(QtCore.Qt.ApplicationModal)
        ok = QtWidgets.QPushButton('OK',self.widgets.about)
        ok.clicked.connect(self.widgets.about.accept)
        ok.setDefault(True)
        self.widgets.about.layout = QtWidgets.QGridLayout(self.widgets.about)
        self.widgets.about.layout.addWidget(QtWidgets.QLabel(txt),0,0,1,5)
        self.widgets.about.layout.addWidget(ok,1,1)
        self.widgets.about.exec_()

    def _InitParams(self):
        params = [
                {'name': 'Picoharp', 'type': 'group', 'children':[
                        {'name': 'tint','type': 'float', 'value': 100e-3, 'step': 1e-3, 'siPrefix': True, 'suffix': 's'},
                        ]
                },
                {'name': 'Hydraharp', 'type': 'group', 'children':[
                        {'name': 'tint','type': 'float', 'value': 100e-3, 'step': 1e-3, 'siPrefix': True, 'suffix': 's'},
                        ]
                },                
                {'name': 'Perkin Elmer', 'type': 'group', 'children':[
                        {'name': 'Channel','type': 'str', 'value': 'Dev1/ctr0'},                        
                        {'name': 'tint','type': 'float', 'value': 20e-3, 'step': 1e-3, 'siPrefix': True, 'suffix': 's'},
                        ]
                },
                {'name': 'Analog In', 'type': 'group', 'children':[
                        {'name': 'Channel','type': 'str', 'value': 'Dev1/ai1'},
                        {'name': 'tint','type': 'float', 'value': 20e-3, 'step': 1e-3, 'siPrefix': True, 'suffix': 's'},
                        {'name': 'Vmax', 'type': 'float', 'value': 10, 'step': 1e-3, 'siPrefix': True, 'suffix': 'V'},
                        {'name': 'Vmin', 'type': 'float', 'value': 0, 'step': 1e-3, 'siPrefix': True, 'suffix': 'V'},
                        ]
                },
                {'name': 'CCD', 'type': 'group', 'children':[
                        ]
                },                
                {'name': 'Spectrometer', 'type': 'group', 'children':[
                        {'name': 'tint','type': 'float', 'value': .1, 'step': .1, 'siPrefix': True, 'suffix': 's'},
                        {'name': 'Name Server','type': 'list', 'values': {'qpg-lab1-pc.dhcp.phys.ethz.ch',
                                                                          'g13-winspec-pc.dhcp.phys.ethz.ch',
                                                                          }, 'value': 'qpg-lab1-pc.dhcp.phys.ethz.ch'
                        },
                        ]
                },            
                {'name': 'Powermeter', 'type': 'group', 'children':[
                        {'name': 'tint','type': 'float', 'value': 100e-3, 'step': 1e-3, 'siPrefix': True, 'suffix': 's'},
                        ]
                },                  
                ]
        self.params = Parameter.create(name='params',type = 'group', children=params)
        
    def _settings(self):
        self.widgets.settings = QtWidgets.QDialog()
        self.widgets.settings.setWindowTitle('Parameter Settings')
        accept = QtWidgets.QPushButton('Set', self.widgets.settings)
        accept.clicked.connect(self._settingschange)
        close = QtWidgets.QPushButton('Close', self.widgets.settings)
        close.clicked.connect(self.widgets.settings.accept)
        self.widgets.paramtree = ParameterTree()
        self.widgets.paramtree.setParameters(self.params, showTop=False)
        self.widgets.settings.layout = QtWidgets.QGridLayout(self.widgets.settings)
        self.widgets.settings.layout.addWidget(self.widgets.paramtree)
        self.widgets.settings.layout.addWidget(accept)
        self.widgets.settings.layout.addWidget(close)
        self.widgets.settings.show()
        self.widgets.settings.resize(350,520)

    def _settingschange(self):
        running = self.threads.__dict__.items()
        [getattr(self.actions,key).setChecked(False) for key, thread in running
            if not key.startswith('__') if not thread == None
        ]
        [getattr(self.actions,key).setChecked(True) for key, thread in running
            if not key.startswith('__') if not thread == None
        ]         
        
    def _restore(self):
        self.area.restoreState(self.state)
           
    def _GenWindow(self):
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(1024,800)
        self.showMaximized()
        self.setWindowTitle('G8 Lab Suite')
        
    def _GenStatusbar(self):
        self.statusBar()
        
    def _GenMenuBar(self):
        self.menus = Object()
        self.menus.bar = self.menuBar()
        self.menus.file = self.menus.bar.addMenu('&File')
        self.menus.file.addAction(self.actions.settings)        
        self.menus.file.addAction(self.actions.restore)
        self.menus.file.addAction(self.actions.exit)
        self.menus.devices = self.menus.bar.addMenu('&Devices')
        self.menus.devices.addAction(self.actions.pico)
        self.menus.devices.addAction(self.actions.hydra)
        self.menus.devices.addAction(self.actions.pe)
        self.menus.devices.addAction(self.actions.ai)
        self.menus.devices.addAction(self.actions.ccd)
        self.menus.devices.addAction(self.actions.ws)
        self.menus.devices.addAction(self.actions.pm)
        self.menus.help = self.menus.bar.addMenu('&Help')
        self.menus.help.addAction(self.actions.about)    
        
    def _GenToolBar(self):
        self.toolbar = Object()
        self.toolbar.cfg = self.addToolBar('Config')
        self.toolbar.cfg.setMovable(False)
        self.toolbar.cfg.addAction(self.actions.exit)
        self.toolbar.cfg.addAction(self.actions.settings)    
        self.toolbar.cfg.addAction(self.actions.restore)
        self.toolbar.dev = self.addToolBar('Devices')
        self.toolbar.dev.setMovable(False)
        self.toolbar.dev.addAction(self.actions.pico)
        self.toolbar.dev.addAction(self.actions.hydra)
        self.toolbar.dev.addAction(self.actions.pe)
        self.toolbar.dev.addAction(self.actions.ai)
        self.toolbar.dev.addAction(self.actions.ccd)
        self.toolbar.dev.addAction(self.actions.ws)
        self.toolbar.dev.addAction(self.actions.pm)
        
    def _GenActions(self):
        self.actions = Object()
        self.actions.exit = QtWidgets.QAction(QtGui.QIcon('./icons/exit.png'), 'Exit', self)
        self.actions.exit.setShortcut('Ctrl+Q')
        self.actions.exit.setStatusTip('Exit Application')
        self.actions.exit.triggered.connect(self.close)        
        self.actions.about = QtWidgets.QAction('About', self)
        self.actions.about.setStatusTip('Show author information')
        self.actions.about.triggered.connect(self._showabout)         
        self.actions.pico = QtWidgets.QAction(QtGui.QIcon('./icons/pico.png'),'Picoharp', self, checkable=True)
        self.actions.pico.setStatusTip('Toggle Picoharp APDs on / off')
        self.actions.pico.toggled.connect(self._statPICO)
        self.actions.hydra = QtWidgets.QAction(QtGui.QIcon('./icons/hydra.png'),'Hydraharp', self, checkable=True)
        self.actions.hydra.setStatusTip('Toggle Hydraharp APDs on / off')
        self.actions.hydra.toggled.connect(self._statHYDRA)        
        self.actions.pe = QtWidgets.QAction(QtGui.QIcon('./icons/excelitas.png'),'Perkin Elmer', self, checkable=True)
        self.actions.pe.setStatusTip('Toggle Perkin Elmer APDs on / off')
        self.actions.pe.toggled.connect(self._statPE)              
        self.actions.ai = QtWidgets.QAction(QtGui.QIcon('./icons/ai.png'),'Analog In', self, checkable=True)
        self.actions.ai.setStatusTip('Toggle Analog In Voltage on / off')
        self.actions.ai.toggled.connect(self._statAI)    
        self.actions.ccd = QtWidgets.QAction(QtGui.QIcon('./icons/ccd.png'),'CCD', self, checkable=True)
        self.actions.ccd.setStatusTip('Toggle CCD on / off')
        self.actions.ccd.toggled.connect(self._statCCD)
        self.actions.ws = QtWidgets.QAction(QtGui.QIcon('./icons/ws.png'),'WinSpec', self, checkable=True)
        self.actions.ws.setStatusTip('Toggle WinSpec on / off')
        self.actions.ws.toggled.connect(self._statWS)
        self.actions.pm = QtWidgets.QAction(QtGui.QIcon('./icons/pm.png'),'Powermeter', self, checkable=True)
        self.actions.pm.setStatusTip('Toggle Powermeter on / off')
        self.actions.pm.toggled.connect(self._statPM)
        self.actions.settings = QtWidgets.QAction(QtGui.QIcon('./icons/settings.png'), 'Settings', self)
        self.actions.settings.setStatusTip('Configure settings')
        self.actions.settings.triggered.connect(self._settings)
        self.actions.restore = QtWidgets.QAction(QtGui.QIcon('./icons/restore.png'), 'Restore', self)
        self.actions.restore.setStatusTip('Restore Window to Defaults')
        self.actions.restore.triggered.connect(self._restore)
        
    def _GenThreads(self):
        self.threads = Object()
        self.threads.pico   = None
        self.threads.hydra  = None
        self.threads.pe     = None
        self.threads.ai     = None
        self.threads.ccd    = None
        self.threads.ws     = None
        self.threads.pm     = None
        
    def _GenDocks(self):
        self.docks = Object()
        self.docks.pico = Dock("Picoharp APD Counts", size=(1,1))
        self.docks.hydra= Dock("Hydraharp APD Counts", size=(1,1))
        self.docks.pe   = Dock("PE APD Counts", size=(1,1))
        self.docks.ai   = Dock("Analog In Voltage", size=(1,1))
        self.docks.ccd  = Dock("CCD Image", size=(1,1))
        self.docks.ws   = Dock("Spectrometer", size=(1,1))
        self.docks.pm   = Dock("Powermeter", size=(1,1))
        self.docks.cons = Dock("Console", size=(1,1))
        self.area.addDock(self.docks.pico,  'left')
        self.area.addDock(self.docks.hydra,  'left')
        self.area.addDock(self.docks.pe,  'left')
        self.area.addDock(self.docks.ai,   'left')
        self.area.addDock(self.docks.ccd, 'left')
        self.area.addDock(self.docks.ws,   'left')
        self.area.addDock(self.docks.pm,   'left')
        self.area.addDock(self.docks.cons,   'left')        
        self.area.moveDock(self.docks.pm,'above',self.docks.cons)         
        self.area.moveDock(self.docks.ws,'above',self.docks.pm)        
        self.area.moveDock(self.docks.ccd,'above',self.docks.ws)
        self.area.moveDock(self.docks.ai,'above',self.docks.ccd)
        self.area.moveDock(self.docks.pe,'above',self.docks.ai)
        self.area.moveDock(self.docks.hydra,'above',self.docks.pe)
        self.area.moveDock(self.docks.pico,'above',self.docks.hydra)
        self.state = self.area.saveState()

    def _statPICO(self):
        if self.actions.pico.isChecked():
            self.threads.pico = thread_pico(tint=self.params.param('Picoharp').param('tint').value())
            self.threads.pico.new_data.connect(self._updatePICO)
            self.threads.pico.start()
        if self.threads.pico.isRunning() and not self.actions.pico.isChecked():
            self.threads.pico.stop()
            
    def _updatePICO(self,data):
        self.buffers.pico1.append(data[0])
        self.y.pico1[:] = self.buffers.pico1
        self.plots.pico1.setData(self.x.pico,self.y.pico1)
        self.hl.pico1.setPos(self.y.pico1[-1])
        self.buffers.pico2.append(data[1])
        self.y.pico2[:] = self.buffers.pico2
        self.plots.pico2.setData(self.x.pico,self.y.pico2)
        self.hl.pico2.setPos(self.y.pico2[-1])
        self.text.setText('Ratio %.1e' % np.nan_to_num(self.y.pico1[-1]/(self.y.pico2[-1])))        

    def _statHYDRA(self):
        if self.actions.hydra.isChecked():
            self.threads.hydra = thread_hydra(tint=self.params.param('Hydraharp').param('tint').value())
            self.threads.hydra.new_data.connect(self._updateHYDRA)
            self.threads.hydra.start()
        if self.threads.hydra.isRunning() and not self.actions.hydra.isChecked():
            self.threads.hydra.stop()
            
    def _updateHYDRA(self,data):
        self.buffers.hydra1.append(data[0])
        self.y.hydra1[:] = self.buffers.hydra1
        self.plots.hydra1.setData(self.x.hydra,self.y.hydra1)
        self.hl.hydra1.setPos(self.y.hydra1[-1])
        self.buffers.hydra2.append(data[1])
        self.y.hydra2[:] = self.buffers.hydra2
        self.plots.hydra2.setData(self.x.hydra,self.y.hydra2)
        self.hl.hydra2.setPos(self.y.hydra2[-1])
        self.buffers.hydra3.append(data[2])
        self.y.hydra3[:] = self.buffers.hydra3
        self.plots.hydra3.setData(self.x.hydra,self.y.hydra3)
        self.hl.hydra3.setPos(self.y.hydra3[-1])
        self.text.setText('Ratio %.1e' % np.nan_to_num(self.y.hydra1[-1]/(self.y.hydra2[-1])))        
        
    def _statPE(self):
        if self.actions.pe.isChecked():
            self.threads.pe = thread_pe(chan=self.params.param('Perkin Elmer').param('Channel').value(),
                                        tint=self.params.param('Perkin Elmer').param('tint').value())
            self.threads.pe.new_data.connect(self._updatePE)
            self.threads.pe.start()
        if self.threads.pe.isRunning() and not self.actions.pe.isChecked():
            self.threads.pe.stop()
            
    def _updatePE(self,data):
        self.buffers.pe.append(data)
        self.y.pe[:] = self.buffers.pe
        self.plots.pe.setData(self.x.pe,self.y.pe)
        self.hl.pe.setPos(self.y.pe[-1])
    
    def _statAI(self):
        if self.actions.ai.isChecked():
            self.threads.ai = thread_ai(chan=self.params.param('Analog In').param('Channel').value(),
                                        tint=self.params.param('Analog In').param('tint').value(),
                                        minv=self.params.param('Analog In').param('Vmin').value(),
                                        maxv=self.params.param('Analog In').param('Vmax').value(),
                                        )
            self.threads.ai.new_data.connect(self._updateAI)
            self.threads.ai.start()
        if self.threads.ai.isRunning() and not self.actions.ai.isChecked():
            self.threads.ai.stop()
            
    def _updateAI(self,data):
        self.buffers.ai.append(data)
        self.y.ai[:] = self.buffers.ai
        self.plots.ai.setData(self.x.ai,self.y.ai)
        self.hl.ai.setPos(self.y.ai[-1])      

    def _statCCD(self):
        if self.actions.ccd.isChecked():
            self.threads.ccd = thread_ccd()
            self.threads.ccd.new_data.connect(self._updateCCD)
            self.threads.ccd.start()
        if self.threads.ccd.isRunning() and not self.actions.ccd.isChecked():
            self.threads.ccd.stop()
            
    def _updateCCD(self,data):
        self.widgets.ccd.setImage(data,
                                  autoLevels=False,
                                  autoHistogramRange=False,
                                  levels=[0,2**8-1],
                                  autoRange=False)
                                  
    def _statWS(self):
        if self.actions.ws.isChecked():
            self.threads.ws = thread_ws(tint=self.params.param('Spectrometer').param('tint').value(),
                                        ns=self.params.param('Spectrometer').param('Name Server').value())
            self.threads.ws.new_data.connect(self._updateWS)
            self.threads.ws.start()
        if self.threads.ws.isRunning() and not self.actions.ws.isChecked():
            self.threads.ws.stop()
            
    def _updateWS(self,data):
        self.x.ws = data[0]
        self.y.ws = data[1]
        self.plots.ws.setData(self.x.ws,self.y.ws)
        
    def _statPM(self):
        if self.actions.pm.isChecked():
            self.threads.pm = thread_pm(tint=self.params.param('Powermeter').param('tint').value())
            self.threads.pm.new_data.connect(self._updatePM)
            self.threads.pm.start()
        if self.threads.pm.isRunning() and not self.actions.pm.isChecked():
            self.threads.pm.stop()
            
    def _updatePM(self,data):
        self.buffers.pm.append(data)
        self.y.pm[:] = self.buffers.pm
        self.plots.pm.setData(self.x.pm,self.y.pm)
        self.hl.pm.setPos(self.y.pm[-1])               
        
    def _GenPlotWidgets(self):
        self.widgets = Object()
        self.plots = Object()
        self.hl = Object()
        self._GenPICOWidget()
        self._GenHYDRAWidget()
        self._GenPEWidget()
        self._GenAIWidget()
        self._GenCCDWidget()
        self._GenWSWidget()
        self._GenPMWidget()
        self._GenConsWidget()
        
    def _GenPICOWidget(self):
        self.widgets.pico = pg.PlotWidget(title='Count Rate Picoharp')
        self.widgets.pico.showGrid(x=True,y=True)
        self.widgets.pico.setLabel('left','Counts','Hz')
        self.widgets.pico.setLabel('bottom','time','s')
        self.widgets.pico.addLegend()        
        self.x.pico = np.linspace(-10,0,100)
        self.y.pico1 = np.zeros(100,dtype=np.float)
        self.y.pico2 = np.zeros(100,dtype=np.float)        
        self.plots.pico1 = self.widgets.pico.plot(self.x.pico,self.y.pico1,
                                                  pen=pg.mkPen(color=(255,127,80),width=3),
                                                  name='Ch0')
        self.plots.pico2 = self.widgets.pico.plot(self.x.pico,self.y.pico2,
                                                  pen=pg.mkPen(color=(100,149,237),width=3),
                                                  name='Ch1')
        self.hl.pico1 = pg.InfiniteLine(angle=0, movable = False,
                                        pen=pg.mkPen(color=(255,127,80),style=QtCore.Qt.DashLine))
        self.hl.pico2 = pg.InfiniteLine(angle=0, movable = False,
                                        pen=pg.mkPen(color=(100,149,237),style=QtCore.Qt.DashLine))
        self.widgets.pico.addItem(self.hl.pico1, ignoreBound=True)
        self.widgets.pico.addItem(self.hl.pico2, ignoreBound=True)     
        self.text = pg.TextItem('Ratio 0', anchor=(-0.3,1.3), border='w', fill=(0, 0, 255, 100))
        self.text.setPos(-10,1)
        self.widgets.pico.addItem(self.text)        
        
    def _GenHYDRAWidget(self):
        self.widgets.hydra = pg.PlotWidget(title='Count Rate Hydraharp')
        self.widgets.hydra.showGrid(x=True,y=True)
        self.widgets.hydra.setLabel('left','Counts','Hz')
        self.widgets.hydra.setLabel('bottom','time','s')
        self.widgets.hydra.addLegend()        
        self.x.hydra = np.linspace(-10,0,100)
        self.y.hydra1 = np.zeros(100,dtype=np.float)
        self.y.hydra2 = np.zeros(100,dtype=np.float)     
        self.y.hydra3 = np.zeros(100,dtype=np.float)  
        self.plots.hydra1 = self.widgets.hydra.plot(self.x.hydra,self.y.hydra1,
                                                  pen=pg.mkPen(color=(255,127,80),width=3),
                                                  name='Ch0')
        self.plots.hydra2 = self.widgets.hydra.plot(self.x.hydra,self.y.hydra2,
                                                  pen=pg.mkPen(color=(100,149,237),width=3),
                                                  name='Ch1')
        self.plots.hydra3 = self.widgets.hydra.plot(self.x.hydra,self.y.hydra3,
                                          pen=pg.mkPen(color=(85,180,70),width=3),
                                          name='Sync')
        self.hl.hydra1 = pg.InfiniteLine(angle=0, movable = False,
                                        pen=pg.mkPen(color=(255,127,80),style=QtCore.Qt.DashLine))
        self.hl.hydra2 = pg.InfiniteLine(angle=0, movable = False,
                                        pen=pg.mkPen(color=(100,149,237),style=QtCore.Qt.DashLine))
        self.hl.hydra3 = pg.InfiniteLine(angle=0, movable = False,
                                        pen=pg.mkPen(color=(85,180,70),style=QtCore.Qt.DashLine))
        self.widgets.hydra.addItem(self.hl.hydra1, ignoreBound=True)
        self.widgets.hydra.addItem(self.hl.hydra2, ignoreBound=True)    
        self.widgets.hydra.addItem(self.hl.hydra3, ignoreBound=True)  
        self.text = pg.TextItem('Ratio 0', anchor=(-0.3,1.3), border='w', fill=(0, 0, 255, 100))
        self.text.setPos(-10,1)
        self.widgets.hydra.addItem(self.text)           
        
    def _GenPEWidget(self):
        self.widgets.pe = pg.PlotWidget(title='Count Rate Perkin Elmer APD')
        self.widgets.pe.showGrid(x=True,y=True)
        self.widgets.pe.setLabel('left','Counts','Hz')
        self.widgets.pe.setLabel('bottom','time','s')
        self.widgets.pe.addLegend()        
        self.x.pe = np.linspace(-10,0,250)
        self.y.pe = np.zeros(250,dtype=np.float)
        self.plots.pe = self.widgets.pe.plot(self.x.pe,self.y.pe,pen=(255,0,0))

        self.hl.pe = pg.InfiniteLine(angle=0, movable = False,
                                     pen=pg.mkPen(color=(255,255,255),style=QtCore.Qt.DashLine))
        self.widgets.pe.addItem(self.hl.pe, ignoreBound=True)
        
    def _GenAIWidget(self):
        self.widgets.ai = pg.PlotWidget(title='Analog In Voltage')
        self.widgets.ai.showGrid(x=True,y=True)
        self.widgets.ai.setLabel('left','Voltage','V')
        self.widgets.ai.setLabel('bottom','time','s')
        self.x.ai = np.linspace(-10,0,250)
        self.y.ai = np.zeros(250,dtype=np.float)
        self.plots.ai = self.widgets.ai.plot(self.x.ai,self.y.ai,pen=(255,0,0))
        self.hl.ai = pg.InfiniteLine(angle=0, movable = False,
                                     pen=pg.mkPen(color=(255, 255, 255),style=QtCore.Qt.DashLine))
        self.widgets.ai.addItem(self.hl.ai, ignoreBound=True)

    def _GenCCDWidget(self):
        self.widgets.ccd = pg.ImageView()
        cmap = pg.ColorMap([0,.5,1],cm.Blues_r([0,127,255])[:,:-1])
        self.widgets.ccd.ui.histogram.gradient.setColorMap(cmap)
        self.widgets.ccd.setImage(np.zeros((1288,964),dtype=np.uint8),
                                  autoLevels=False,
                                  autoHistogramRange=False,
                                  levels=[0,2**8-1],
                                  autoRange=False
                                  )
        
    def _GenWSWidget(self):
        self.widgets.ws = pg.PlotWidget(title='Spectrometer')
        self.widgets.ws.showGrid(x=True,y=True)
        self.widgets.ws.setLabel('left','Intensity','Counts')
        self.widgets.ws.setLabel('bottom','Wavelength','m')
        self.x.ws = np.arange(1340)
        self.y.ws = np.zeros(1340,dtype=np.uint16)  
        self.plots.ws = self.widgets.ws.plot(self.x.ws,self.y.ws,pen=(255,0,0))
        
    def _GenPMWidget(self):
        self.widgets.pm = pg.PlotWidget(title='Powermeter Readout')
        self.widgets.pm.showGrid(x=True,y=True)
        self.widgets.pm.setLabel('left','Power','W')
        self.widgets.pm.setLabel('bottom','time','s')
        self.x.pm = np.linspace(-10,0,100)
        self.y.pm = np.zeros(100,dtype=np.float)
        self.plots.pm = self.widgets.pm.plot(self.x.pm,self.y.pm,pen=(255,0,0))
        self.hl.pm = pg.InfiniteLine(angle=0, movable = False,
                                     pen=pg.mkPen(color=(255, 255, 255),style=QtCore.Qt.DashLine))
        self.widgets.pm.addItem(self.hl.pm, ignoreBound=True)        
            
    def _GenConsWidget(self):
        namespace = {'np': np}
        text = """
            This is an interactive python console. The numpy module has already been imported as 'np'. 
            """
        self.widgets.cons = ConsoleWidget(namespace = namespace, text = text)
#        self.widgets.cons.output.setStyleSheet('color: white; background-color: black;')

    def _DockWidgets(self):
        self.docks.pico.addWidget(self.widgets.pico)
        self.docks.hydra.addWidget(self.widgets.hydra)
        self.docks.pe.addWidget(self.widgets.pe)
        self.docks.ai.addWidget(self.widgets.ai)
        self.docks.ccd.addWidget(self.widgets.ccd)
        self.docks.ws.addWidget(self.widgets.ws)
        self.docks.pm.addWidget(self.widgets.pm)
        self.docks.cons.addWidget(self.widgets.cons)
        
    def _GenData(self):
        self.buffers = Object()
        self.buffers.pico1 = collections.deque([0.0]*100, 100)
        self.buffers.pico2 = collections.deque([0.0]*100, 100)
        self.buffers.hydra1 = collections.deque([0.0]*100, 100)
        self.buffers.hydra2 = collections.deque([0.0]*100, 100) 
        self.buffers.hydra3 = collections.deque([0.0]*100, 100) 
        self.buffers.pe = collections.deque([0.0]*250, 250)
        self.buffers.ai  = collections.deque([0,0]*250, 250)
        self.buffers.pm  = collections.deque([0,0]*100, 100)
        self.x = Object()
        self.y = Object()
        
    def _SetRangePlots(self):
        self.widgets.pico.plotItem.vb.setRange = self._setRangePICO
        self.widgets.hydra.plotItem.vb.setRange = self._setRangeHYDRA
        self.widgets.pe.plotItem.vb.setRange = self._setRangePE
        self.widgets.ai.plotItem.vb.setRange  = self._setRangeAI
        self.widgets.pm.plotItem.vb.setRange  = self._setRangePM
        
    def _setRangePICO(self, rect=None, xRange=None, yRange=None, *args, **kwds):
        if not kwds.get('disableAutoRange', True):
            if yRange is not None:
                yRange[0] = 0
        pg.ViewBox.setRange(self.widgets.pico.plotItem.vb, rect, xRange, yRange, *args, **kwds)

    def _setRangeHYDRA(self, rect=None, xRange=None, yRange=None, *args, **kwds):
        if not kwds.get('disableAutoRange', True):
            if yRange is not None:
                yRange[0] = 0
        pg.ViewBox.setRange(self.widgets.hydra.plotItem.vb, rect, xRange, yRange, *args, **kwds)
        
    def _setRangePE(self, rect=None, xRange=None, yRange=None, *args, **kwds):
        if not kwds.get('disableAutoRange', True):
            if yRange is not None:
                yRange[0] = 0
        pg.ViewBox.setRange(self.widgets.pe.plotItem.vb, rect, xRange, yRange, *args, **kwds)
           
    def _setRangeAI(self, rect=None, xRange=None, yRange=None, *args, **kwds):
        if not kwds.get('disableAutoRange', True):
            if yRange is not None:
                yRange[0] = 0
        pg.ViewBox.setRange(self.widgets.ai.plotItem.vb, rect, xRange, yRange, *args, **kwds)      

    def _setRangePM(self, rect=None, xRange=None, yRange=None, *args, **kwds):
        if not kwds.get('disableAutoRange', True):
            if yRange is not None:
                yRange[0] = 0
        pg.ViewBox.setRange(self.widgets.pm.plotItem.vb, rect, xRange, yRange, *args, **kwds)      
        
            
class thread_pico(QtCore.QThread):

    new_data = QtCore.pyqtSignal(object)
    
    def __init__(self,tint):
        QtCore.QThread.__init__(self)
        self.running = True
        self.apd = picoharp()
        self.apd.opendefaults()
        self.tint = tint
        self.nint = tint/100e-3
        
    def __del__(self):
        self.wait()
        
    def run(self):
        if self.nint >= 1.:
            while self.running:
                cts0 = 0
                cts1 = 0
                for _ in range(np.round(self.nint).astype(int)):
                    self.msleep(100)
                    cts0 += self.apd.GetCountRate(chan=0)
                    cts1 += self.apd.GetCountRate(chan=1)
                self.new_data.emit((cts0/self.nint,cts1/self.nint))
        else:      
            while self.running:
                self.msleep(self.tint*1e3)
                self.new_data.emit((self.apd.GetCountRate(chan=0),self.apd.GetCountRate(chan=1)))
        
    def stop(self):
        self.running = False
        self.wait()
        self.apd.CloseDevice()
        self.terminate()


class thread_hydra(QtCore.QThread):

    new_data = QtCore.pyqtSignal(object)
    
    def __init__(self,tint):
        QtCore.QThread.__init__(self)
        self.running = True
        self.apd = hydraharp()
        self.apd.OpenDevice()
        self.apd.Initialize()
        # WE REALLY DO NOT WANT TO CHANGE DEVICE SETTINGS HERE... 
        # I comment this out (PK)
        #self.apd.SetInputCFD(0,260,20)
        #self.apd.SetInputCFD(1,252,20)
        #self.apd.SetSyncDiv(1)
        #self.apd.SetSyncChannelOffset(-55000) 
        self.tint = tint

    def __del__(self):
        self.wait()
        
    def run(self):
        while self.running:
            self.msleep(self.tint*1e3)
            self.new_data.emit((self.apd.GetCountRate(channel=0),self.apd.GetCountRate(channel=1),self.apd.GetSyncRate()))
        
    def stop(self):
        self.running = False
        self.wait()
        self.apd.CloseDevice()
        self.terminate()

        
class thread_pe(QtCore.QThread):

    new_data = QtCore.pyqtSignal(object)
    
    def __init__(self,chan,tint):
        QtCore.QThread.__init__(self)
        self.running = True
        self.tint = tint
        self.chan = chan
        
    def __del__(self):
        self.wait()
        
    def run(self):
        with nidaqmx.Task() as master_task, nidaqmx.Task() as slave_task:
            master_task.ai_channels.add_ai_voltage_chan('Dev1/ai0')
            slave_task.ci_channels.add_ci_count_edges_chan(self.chan)
            #slave_task.ci_channels.add_ci_count_edges_chan('Dev1/ctr3')
            master_task.timing.cfg_samp_clk_timing(1e3, sample_mode=AcquisitionType.CONTINUOUS,\
                                                   samps_per_chan=int(self.tint*1e3))
            slave_task.timing.cfg_samp_clk_timing( 1e3, sample_mode=AcquisitionType.CONTINUOUS,\
                                                  samps_per_chan=int(self.tint*1e3), source="/Dev1/ai/SampleClock")
            self.ci = slave_task
            self.co = master_task
            
            self.ci.start()
            self.co.start()
            while self.running:
                data = self.ci.read(int(self.tint*1e3))
                self.new_data.emit((data[-1]-data[0])/self.tint)
            self.co.stop()
            self.ci.stop()
        
    def stop(self):
        self.running = False
        self.wait()
        self.terminate()        

            
class thread_ai(QtCore.QThread):

    new_data = QtCore.pyqtSignal(object)
    
    def __init__(self,chan,tint,minv,maxv):
        QtCore.QThread.__init__(self)
        self.running = True
        self.tint    = tint
        self.chan = chan #
        self.minv = minv #
        self.maxv = maxv #
#        self.ai = AnalogInputTask()
#        self.ai.create_voltage_channel(chan,
#                                       terminal = 'diff',
#                                       min_val = minv,
#                                       max_val = maxv
#                                       )
##        self.ai.configure_timing_sample_clock(rate = int(1e4),
##                                              sample_mode = 'continuous',
##                                              samples_per_channel = int(1e1)
##                                              )
    def __del__(self):
        self.wait()
        
    def run(self):
        while self.running:
            with nidaqmx.Task() as task:
#                task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
                in_channel = task.ai_channels.add_ai_voltage_chan(self.chan)
                #in_channel.ai_rng_high = self.maxv
                #in_channel.ai_rng_low = self.minv
                self.ai = task
                self.msleep(self.tint*1e3)
                self.new_data.emit(self.ai.read())
        
    def stop(self):
        self.running = False
        self.wait()
        self.terminate()
        

class thread_ccd(QtCore.QThread):

    new_data = QtCore.pyqtSignal(object)
    
    def __init__(self):
        QtCore.QThread.__init__(self)
        self.running = True
        self.ccd = pc.Camera()
        self.ccd.connect(pc.BusManager().getCameraFromIndex(0))
        self.caminfo = self.ccd.getCameraInfo()
        self.res = [int(val) for val in self.caminfo.sensorResolution.split(b'x')]
        self.ccd.startCapture()
        self.msleep(1000)

    def __del__(self):
        self.wait()
        
    def run(self):
        while self.running:
            self.msleep(50)
            try:
                data = self.ccd.retrieveBuffer()
                self.new_data.emit(np.array(data.getData(),dtype=np.uint8).reshape(self.res[1],self.res[0]).T)
            except pc.Fc2error as fc2err:
                print("Error retrieving buffer : ", fc2err)
                continue
        self.msleep(100)
        self.ccd.stopCapture()
        self.msleep(1000)
        self.ccd.disconnect()    
        
    def stop(self):
        self.running = False
        self.wait()
        self.terminate()         
        

class thread_ws(QtCore.QThread):

    new_data = QtCore.pyqtSignal(object)
    
    def __init__(self,tint,ns):
        
        self.nsdic = {
            'qpg-lab1-pc.dhcp.phys.ethz.ch': ('photon1', 'G8_Winspec'),
            'g13-winspec-pc.dhcp.phys.ethz.ch': ('G13_WinSpec', 'G13_Winspec'),
            }
        QtCore.QThread.__init__(self)
        self.running = True
        NS = Pyro4.naming.locateNS(host=ns,hmac_key=self.nsdic.get(ns)[0])
        self.ws = Pyro4.Proxy(NS.lookup(self.nsdic.get(ns)[1]))
        self.ws.num_frames = 1
        self.ws.exposure_time = tint
        self.wl = np.array(self.ws.get_spectrum(wlen=True)[1])*1e-9

    def __del__(self):
        self.wait()
        
    def run(self):
        while self.running:
            self.msleep(int(self.ws.exposure_time*1e3))
            self.new_data.emit([self.wl,np.array(self.ws.get_spectrum(),dtype=np.uint16)[0,:,0]])
        
    def stop(self):
        self.running = False
        self.wait()
        self.msleep(20)
        self.terminate()          


class thread_pm(QtCore.QThread):

    new_data = QtCore.pyqtSignal(object)
    
    def __init__(self,tint):
        QtCore.QThread.__init__(self)
        self.running = True
        self.pm = ThorlabsPM100(avg=1)
        self.tint = tint
        
    def __del__(self):
        self.wait()
        
    def run(self):
        while self.running:
            self.pm._wait()
            self.new_data.emit(self.pm.read)
        
    def stop(self):
        self.running = False
        self.wait()
        self.msleep(self.tint*1e3)
        self.pm.close()
        self.terminate()

        
class Object(object):
    pass


def run():
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('./icons/crown.png'))
    win = LivePlotter()
    win.show()
    sys.exit(app.exec_())
        
    
def create_window(window_class):
    app_created = False
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtCore.QApplication(sys.argv)
        app_created = True
    app.references = set()
    window = window_class()
    app.references.add(window)
    window.show()
    if app_created:
        app.exec_()
    return window
        
    
if __name__ == '__main__':
    run()
#    win = create_window(LivePlotter)