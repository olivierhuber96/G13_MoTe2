import pyqtgraph as pg
import pyqtgraph.multiprocess as pgmp
from pyqtgraph.Qt import QtGui, QtCore


class LivePlot():
    _proc = None
    _rpg = None

    @classmethod
    def _init_qt(cls):
        pg.mkQApp()
        cls._proc = pgmp.QtProcess()
        cls._rpg = cls._proc._import('pyqtgraph')
        cls._rpg.setConfigOptions(antialias=True)

    def __init__(self, figsize=(800, 600), x_label='', y_label=''):
        if self._proc is None:
            self._init_qt()
        self.win = self._rpg.GraphicsWindow(title='Live Plot')
        self.win.setBackground('w')
        self.win.resize(*figsize)
        self.plt = self.win.addPlot()
        self.plt.setLabel('bottom', x_label)
        self.plt.setLabel('left', y_label)


    # Wrapper methods for pyqtgraph object

    def clear(self):
        self.plt.clear()

    def reset(self, ncol=1, nrow=1):
        self.win.clear()
        plt_list = []
        for i in range(nrow):
            for j in range(ncol):
                plt_list.append(self.win.addPlot(row=i, col=j))

        self.plt = plt_list[0]
        return plt_list

    def plot(self, *args, **kwargs):
        self.plt.plot(*args, **kwargs)

    def set_xlabel(self, *args, **kwargs):
        self.plt.setLabel('bottom', *args, **kwargs)

    def set_ylabel(self, *args, **kwargs):
        self.plt.setLabel('left', *args, **kwargs)

    def setLabel(self, *args, **kwargs):
        self.plt.setLabel(*args, **kwargs)

    def close(self):
        self._proc.close()

    def restart_plot(self):
        if self._proc.exited is not True:
            raise RuntimeError('Old process is not exited!')
        self._init_qt()
        self.__init__()

#    def __del__(self):
#        self.close()
