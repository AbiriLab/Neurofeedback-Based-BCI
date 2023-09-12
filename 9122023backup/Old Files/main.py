import UnicornPy
import numpy as np
import csv
import pandas as pd
from datetime import date
from datetime import datetime
import os
import time
import random
from Trial_struct import *
import main_gui
import get_param
import serial
import pygame
import sys
from os.path import exists
from os import walk


class StartRecording:
    def __init__(self, len_trial_rec, eeg_filename, event_filename, test_signal_enabled, trial_number, patient_name,
                 Unicorn_id, task_number):
        self.AcquisitionDurationInSeconds = len_trial_rec
        self.EEG_file_name = eeg_filename
        self.event_file_name = event_filename
        self.numberOfAcquiredChannels = 0
        self.configuration = 0
        self.SamplingRate = 0
        self.FrameLength = 1
        self.device = 0
        self.tests_signal_enabled = test_signal_enabled
        self.patient_name = patient_name
        self.trial = trial_number
        self.unicorn_id = Unicorn_id
        self.task_number = task_number
        self.header = 'FZ, FC1, FC2, C3, CZ, C4, CPZ, PZ, AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ, Battery, ' \
                      'Sample, Unknown '

    def get_device_id(self):
        """gets available eeg devices and return the one you want"""
        # get list of available devices
        # raise error if not available
        # return unicorn device id you want to later pass to eeg_connect
        pass

    def eeg_connect(self):
        """ Connect to EEG device, acquiring channels, config, sampling rate"""
        dev_id = self.get_device_id()
        self.device = UnicornPy.Unicorn(dev_id)
        print("Connected to '%s'." % dev_id)
        self.numberOfAcquiredChannels = self.device.GetNumberOfAcquiredChannels()
        self.configuration = self.device.GetConfiguration()
        self.SamplingRate = UnicornPy.SamplingRate

    def start_acquisition(self, eeg_path, event_path):
        # eeg_path/event_path from make_directory function used to create csv files for recording
        # how program actual records data
        # contains sampling rate adn frame limit calculations
        # gets data from eeg headset
        # write data to paths (csv files)
        # updates gui based on trial number
        # connects to Arduino
        pass

    def make_dir(self, pathname):
        """Creates directories to store collected data using specified pathname"""
        current_work_dir = os.getcwd()
        path = os.path.join(current_work_dir, "EEGdata")

        today = date.today()

        today_date = today.strftime("%Y%m%d")
        path = os.path.join(path, today_date)

        directory = self.patient_name
        path = os.path.join(path, directory)

        directory = self.task_number
        path1 = os.path.join(path, directory)

        # add time to directory
        now = datetime.now()
        current_time = now.strftime("%H%M%S")
        path2 = os.path.join(path1, current_time)

        directory = pathname
        path3 = os.path.join(path2, directory)

        if not os.path.exists(path3):
            os.makedirs(path3)
        return path3, path

    def main(self):
        eeg_path, second_path = self.make_dir("eeg")
        event_path = self.make_dir("event")
        self.eeg_connect()
        self.start_acquisition(eeg_path, event_path)


if __name__ == "__main__":
    data_recording = StartRecording(len_trial_rec=get_param.Trial_duration,
                                    eeg_filename="EEG.csv",
                                    event_filename="Event.csv",
                                    test_signal_enabled=False,
                                    trial_number=get_param.Trial_number,
                                    patient_name=get_param.Username,
                                    Unicorn_id=get_param.Unicorn_ID,
                                    task_number=get_param.task_list)
    data_recording.main()
