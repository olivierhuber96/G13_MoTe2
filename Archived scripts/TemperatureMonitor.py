# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 21:11:30 2017

@author: Thomas Fink, Quantum  Photonics Group, ETH Zurich

The TSP class provides a wrapper to read out a Thorlabs TSP01 temperature sen-
sor via the pyvisa package. The serial number, e.g. 'M00469416', has to be pro-
vided as shown in the exemplary __main__ block at the end of the file.

"""


import pyvisa as visa
import logging

class TSP(object):
    
    def __init__(self,serialnum):
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        try:
#            print(visa.ResourceManager().list_resources())
            port = r'USB0::0x1313::0x80F8::%s::INSTR' % serialnum
            self._s = visa.ResourceManager().open_resource(port)
            _, name, serial, fw = self._s.query("*IDN?")[:-1].split(',')
            self.logger.info('%s %s connected. Firmware version %s.'
                             % (name, serial, fw)
                             )
            self.name   = name
            self.serial = serial
            self.fw     = fw
#            self._s.write("*rst; status:preset; *cls")

        except Exception:
            self.logger.error('Could not connect to serial Port %s' % port)

    def _test(self):
        return self._s.query("*TST?")

    def _reset(self):
        return self._s.query("*RST")

    @property
    def _opc(self):
        return self._s.query("*OPC?")
    @_opc.setter
    def _opc(self):
        return self._s.query("*OPC")

    def close(self):
        self._s.close()
        self.logger.info('TSP01 disconnected.')

    @property
    def temp(self):
        return (float(self._s.query("SENS:TEMP:DATA?")),
                float(self._s.query("SENS3:TEMP:DATA?")),
                )              

    @property
    def humidty(self):
        return float(self._s.query("SENS2:HUM:DATA?"))


if __name__ == "__main__":   
    
    import  sys
    
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) 
    
    tsp = TSP('M00469416')
    print(tsp.temp)
#    print(tsp.humidty)
    tsp.close()