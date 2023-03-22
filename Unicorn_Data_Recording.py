import UnicornPy
import numpy as np
from time import perf_counter

# Connect to the Unicorn device
device = UnicornPy.Unicorn("UN-2021.05.36")

# Get the number of acquired channels and the sampling rate
numberOfAcquiredChannels = device.GetNumberOfAcquiredChannels()
configuration = device.GetConfiguration()
SamplingRate = UnicornPy.SamplingRate

# Set the duration of the data acquisition
AcquisitionDurationInSeconds = 1  # 10 seconds

# Set the frame length
FrameLength = 1  # 1 sample per frame

# Calculate the number of GetData calls required
numberOfGetDataCalls = int(AcquisitionDurationInSeconds * SamplingRate / FrameLength)

# Set up a buffer to hold the received data
receiveBufferBufferLength = FrameLength * numberOfAcquiredChannels * 4  # 4 bytes per float32
receiveBuffer = bytearray(receiveBufferBufferLength)

# Start the data acquisition
device.StartAcquisition(True)

# Loop over the required number of GetData calls

tdata=[]
start_time = perf_counter()
print("start time=",start_time)
for i in range(numberOfGetDataCalls):

    # Receives the configured number of samples from the Unicorn device and writes it to the acquisition buffer.
    device.GetData(FrameLength, receiveBuffer, receiveBufferBufferLength)

    # Convert receive buffer to numpy float array 
    dataa = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * FrameLength)
    data = np.reshape(dataa, (FrameLength, numberOfAcquiredChannels))
    print("Sample Number=", i)
    end_time = perf_counter() 
    print("end time=", end_time)
    print(data)
    tdata.append(data)
device.StopAcquisition()
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Do something with the acquired data
    
# fdata = np.reshape(tdata, (numberOfGetDataCalls, numberOfAcquiredChannels))
# print(fdata)
# print(type(fdata))
# print(fdata.shape)
# Stop the data acquisition












