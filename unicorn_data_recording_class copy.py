import UnicornPy
import numpy as np
from time import perf_counter

# Connect to the Unicorn device
device = UnicornPy.Unicorn("UN-2021.05.36")



class UnicornEEGAcquisition:

    def __init__(self, device, acquired_channels, device_config, acquisition_duration, frame_length):
        self.device = device
        self.acquired_channels = acquired_channels
        self.device_config = device_config
        self.sample_rate = UnicornPy.SamplingRate
        self.acquisition_duration = acquisition_duration
        self.frame_length = frame_length
        
        self.data_calls_num = int(self.acquisition_duration * self.sample_rate / self.frame_length)

        self.buffer_length = self.frame_length * self.acquired_channels * 4  # 4 bytes per float32
        self.receive_buffer = bytearray(self.buffer_length)

        self.tdata = []
    
    
 

    def start_eeg_acquisition(self):
        device.StartAcquisition(True)
        
        # start_time = perf_counter()
        # print("start time=",start_time)
              
        for i in range(self.data_calls_num):
            # Receives the configured number of samples from the Unicorn device and writes it to the acquisition buffer.
            start_time = perf_counter()
            print("start time=",start_time)
            device.GetData(self.frame_length, self.receive_buffer, self.buffer_length)
            end_time = perf_counter() 
            print("end time=", end_time)
            # start_time=end_time
            
            print("Sample Number=", i)
            # Convert receive buffer to numpy float array
            # raw_data = np.frombuffer(self.receive_buffer, dtype=np.float32,
            #                          count=self.acquired_channels * self.frame_length)
            # reshaped_data = np.reshape(raw_data, (self.frame_length, self.acquired_channels))  #self.frame_length
        
            
            
            # print(reshaped_data)
            # self.tdata.append(reshaped_data)
            
            # print(self.tdata)

            # f_data = np.reshape(self.tdata, (self.data_calls_num, self.acquired_channels)) #self.data_calls_num
            # print(f_data)
            # print( self.sample_rate)
            # print(self.acquisition_duration)
            # print(self.data_calls_num)
            # print(self.frame_length)
            # return f_data
            

        device.StopAcquisition()

    def stop_eeg_acquisition(self):
        device.StopAcquisition()
        print('Acquisition Stopped')
        


if __name__ == "__main__":
    eeg_acquisition = UnicornEEGAcquisition(device=UnicornPy.Unicorn("UN-2021.05.36"),
                                            acquired_channels=device.GetNumberOfAcquiredChannels(),
                                            device_config=device.GetConfiguration(),
                                            acquisition_duration=1, frame_length=1)
    eeg_acquisition.start_eeg_acquisition()


