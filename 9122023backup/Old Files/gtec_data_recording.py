import pygds
import numpy as np
import os
import math
from scipy import io
from datetime import datetime

import pandas as pd

# FILENAME = 'RecordedData'
#
# # define acquisition device constants (g.USBamp --> GDS)
# SERIALNUMBER = 'UR-2020.08.14'
# SAMPLINGRATE = 256
# BLOCKSIZE = 8  # block size for GetData (data retrieval from GDS --> Python)
#
# NUMBEROFSCANS = 0  # block size for amplifier (data transmission from device --> GDS)
# # zero for default, values depending on sampling rate
# # integral value; should be an integral multiple of 'BLOCKSIZE' parameter
#
#
# # define visualization constants
# SHOWLASTXSECONDS = 0.004
#
# datetimeStr = datetime.now().strftime("%Y%m%d_%H%M%S")
# filename = FILENAME + '_' + datetimeStr + '.mat'
#
# # samples = []
# os.system('cls')
#
#
# # callback for each GetData cycle when the specified block size of samples was acquired
# def GetDataCallback(dataBlock):
#     samples = []
#     # open the data file and append the latest block of data samples
#     try:
#         file = io.loadmat(filename, variable_names=["data"])
#         samples = file['data']
#     except Exception as e:
#         # Just print(e) is cleaner and more likely what you want,
#         # but if you insist on printing message specifically whenever possible...
#         if hasattr(e, 'message'):
#             print(e.message)
#         else:
#             print(e)
#
#     if len(samples) == 0:
#         samples = np.vstack(dataBlock.copy())
#     else:
#         samples = np.vstack((samples, dataBlock.copy()))
#
#     io.savemat(filename, {"data": samples})
#     # VISUALIZATION OPTION A: show the last X seconds of the data
#     continueDAQ = True
#     totalSamples = samples.shape[0]
#     displaySamples = math.ceil(SHOWLASTXSECONDS * SAMPLINGRATE / BLOCKSIZE) * BLOCKSIZE
#
#     pd.DataFrame(samples).to_excel('my_matrix.xlsx')
#
#     if (totalSamples % displaySamples == 0):
#         samples2display = samples[totalSamples - min(totalSamples, displaySamples):]
#         continueDAQ = scope(samples2display)
#
#         os.system('cls')
#         np.set_printoptions(formatter={'float': '{: 10.3f}'.format}, linewidth=1024)
#         print("\n".join([str(x) for x in samples2display]))
#
#     # VISUALIZATION OPTION B: show only the latest datablock -> does not make sense for little BLOCKSIZE values
#     # continueDAQ = scope(dataBlock)
#
#     return continueDAQ
#
#
# print('Opening device ...   ')
# d = pygds.GDS(SERIALNUMBER)
#
# d.NumberOfScans = NUMBEROFSCANS
# d.SamplingRate = SAMPLINGRATE
# d.CommonGround = [1] * 4
# d.CommonReference = [1] * 4
# d.ShortCutEnabled = 1
# d.CounterEnabled = 0
# d.TriggerEnabled = 1
# for ch in d.Channels:
#     ch.Acquire = 1
#     # set the bandpass filter index -> 45 (0.5 - 30 Hz @ 256 Hz sample rate)
#     ch.BandpassFilterIndex = 45
#     # set the bandpass filter index -> 2 (48 - 52 Hz @ 256 Hz sample rate)
#     # set the bandpass filter index -> 3 (58 - 62 Hz @ 256 Hz sample rate)
#     ch.NotchFilterIndex = 2
#     ch.BipolarChannel = 0
#
# d.SetConfiguration()
#
# # create the visualization scope
# scope = pygds.Scope(1 / SAMPLINGRATE, xlabel='t/s',
#                     ylabel=u'V/μV', title="%s Channels", )
#
# # start and run data acquisition loop (stops automatically when data viewer window is closed)
# data = d.GetData(BLOCKSIZE, GetDataCallback)
#
# d.Close()
# del d


class gtecEEGAcquisition:

    def __int__(self):

        self.FILENAME = 'RecordedData'

        # define acquisition device constants (g.USBamp --> GDS)
        self.SERIALNUMBER = 'UR-2020.08.14'
        self.SAMPLINGRATE = 256
        self.BLOCKSIZE = 8  # block size for GetData (data retrieval from GDS --> Python)

        self.NUMBEROFSCANS = 0  # block size for amplifier (data transmission from device --> GDS)
        # zero for default, values depending on sampling rate
        # integral value; should be an integral multiple of 'BLOCKSIZE' parameter

        # define visualization constants
        self.SHOWLASTXSECONDS = 0.004

        self.datetimeStr = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = self.FILENAME + '_' + self.datetimeStr + '.mat'


        self.d = pygds.GDS(self.SERIALNUMBER)

        self.d.NumberOfScans = self.NUMBEROFSCANS
        self.d.SamplingRate = self.SAMPLINGRATE
        self.d.CommonGround = [1] * 4
        self.d.CommonReference = [1] * 4
        self.d.ShortCutEnabled = 1
        self.d.CounterEnabled = 0
        self.d.TriggerEnabled = 1

        for ch in self.d.Channels:
            ch.Acquire = 1
            # set the bandpass filter index -> 45 (0.5 - 30 Hz @ 256 Hz sample rate)
            ch.BandpassFilterIndex = 45
            # set the bandpass filter index -> 2 (48 - 52 Hz @ 256 Hz sample rate)
            # set the bandpass filter index -> 3 (58 - 62 Hz @ 256 Hz sample rate)
            ch.NotchFilterIndex = 2
            ch.BipolarChannel = 0

        self.d.SetConfiguration()

        self.scope = pygds.Scope(1 / self.SAMPLINGRATE, xlabel='t/s',
                                 ylabel=u'V/μV', title="%s Channels", )

    def GetDataCallback(self, dataBlock):  # idk where dataBlock comes from, if its from pygds or some value we initialize
        samples = []
        # open the data file and append the latest block of data samples
        try:
            file = io.loadmat(self.filename, variable_names=["data"])
            samples = file['data']
        except Exception as e:
            # Just print(e) is cleaner and more likely what you want,
            # but if you insist on printing message specifically whenever possible...
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)

        if len(samples) == 0:
            samples = np.vstack(dataBlock.copy())
        else:
            samples = np.vstack((samples, dataBlock.copy()))

        io.savemat(self.filename, {"data": samples})
        # VISUALIZATION OPTION A: show the last X seconds of the data
        self.continueDAQ = True
        self.totalSamples = samples.shape[0]
        self.displaySamples = math.ceil(self.SHOWLASTXSECONDS * self.SAMPLINGRATE / self.BLOCKSIZE) * self.BLOCKSIZE

        pd.DataFrame(samples).to_excel('my_matrix.xlsx')

        if (self.totalSamples % self.displaySamples == 0):
            samples2display = samples[self.totalSamples - min(self.totalSamples, self.displaySamples):]
            continueDAQ = self.scope(samples2display)

            os.system('cls')
            np.set_printoptions(formatter={'float': '{: 10.3f}'.format}, linewidth=1024)
            print("\n".join([str(x) for x in samples2display]))

        # VISUALIZATION OPTION B: show only the latest datablock -> does not make sense for little BLOCKSIZE values
        # continueDAQ = scope(dataBlock)

        return self.continueDAQ

    # start and run data acquisition loop (stops automatically when data viewer window is closed)

    def start_acquisition_loop(self):
        data = self.d.GetData(self.BLOCKSIZE, self.GetDataCallback())

    def stop_acquisition_loop(self):
        self.d.Close()
        del self.d



