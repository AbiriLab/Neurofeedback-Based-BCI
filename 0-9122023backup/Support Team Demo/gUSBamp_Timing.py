import pygds
import numpy                            #requires numpy 1.15.0 or later
import math
import matplotlib.pyplot as pyplot
from datetime import datetime

# returns a timestamp that can be 
def timestamp():
    return datetime.now()

# returns the number of milliseconds elapsed between the two provided timestamps
def elapsedMilliseconds(startTime, stopTime):
    dt = (stopTime - startTime).total_seconds() * 1000.0
    return dt;

# callback for each GetData cycle when the specified block size of samples was acquired
def retrieveData(block):
    global sampleBlocks, nrBlocks, acquiredSamples, start, avgBlockTime, blockTimes
    time = timestamp()
    if not hasattr(retrieveData, "lastTime"):
        retrieveData.firstTime = time
        retrieveData.lastTime = start
        avgBlockTime = 0
    blockTime = elapsedMilliseconds(retrieveData.lastTime, time)
    blockTimes.append(blockTime)
    totalTime = elapsedMilliseconds(start, time)
    if (retrieveData.firstTime < time):
        avgBlockTime += blockTime / (nrBlocks - 1)
    retrieveData.lastTime = time
    success = sampleBlocks.append(block.copy())
    acquiredSamples += block.shape[0]
    print(f"{acquiredSamples} samples:\t{totalTime:.3f} ms" + (f"\t(dt = {blockTime:.3f} ms)" if len(sampleBlocks) > 1 else "\t(first block includes start of acquisition)"))
    return success or len(sampleBlocks) < nrBlocks

# ---------------------------------------------------------------------

# define acquisition constants for GetData (GDS --> Python)
secondsToAcquire = 5
blockSize = 8

# define acquisition constants for device (g.USBamp --> GDS)
serialNumber = "UR-2020.08.14"
samplingRate = 256
numberOfScans = 0   # zero for default values depending on sampling rate
                    # integral value; should be at least samplingRate * 0.03 sec

"""
if the numberOfScans (d.NumberOfScans) parameter is not an integral multiple of the blockSize parameter used in d.GetData(blockSize),
timing of acquisition may only SEEM to be inaccurate:
- device delivers data in blocks to GDS service, each of them containing a number of d.NumberOfScans scans
- GDS service stores data into buffer
- user retrieves data from GDS buffer (not directly from the device) with d.GetData(n, retrieveData), where n is the number of samples to retrieve and retrieveData is the callback function that is called after n samples have been acquired
- if d.NumberOfScans is not an integral multiple of n, user might need to wait until the device delivers one more block to the GDS service until GetData can return
- this would result in a different timing for the calls to the retrieveData callback function, depending on the number of samples that remain in the buffer since the last call
"""

# ---------------------------------------------------------------------

# open device
d = pygds.GDS(serialNumber)

try:
    # adjust configuration
    d.SamplingRate = samplingRate

    # set block size for acquisition from device (this is not the same as the blockSize for the GetData option)
    # (see detailed comment above)
    if numberOfScans == 0:
        # determine value for d.NumberOfScans parameter automatically
        # defaults for g.Nautilus are 8 (@250 Hz), and 15 (@500 Hz)
        d.NumberOfScans_calc()
    else:
        d.NumberOfScans = numberOfScans
    
    d.CommonGround = [1] * 4
    d.CommonReference = [1] * 4
    d.ShortCutEnabled = 1
    d.CounterEnabled = 1
    d.TriggerEnabled = 0
    for ch in d.Channels:
        ch.Acquire = 1
        ch.BandpassFilterIndex = -1
        ch.NotchFilterIndex = -1
        ch.BipolarChannel = 0
            
    d.SetConfiguration()

    print("Start of acquisition")
    if blockSize % d.NumberOfScans > 0:
        print(f"-> d.NumberOfScans ({d.NumberOfScans}) is not an integral multiple of GetData's blockSize parameter ({blockSize})")
        deviceBlocksPerGetData = blockSize / d.NumberOfScans
        deviceBlockDurationMs = 1000 * d.NumberOfScans / samplingRate
        additionalDeviceBlockDurationMs = math.ceil(deviceBlocksPerGetData) * deviceBlockDurationMs
        intermediateDurationMs = math.floor(deviceBlocksPerGetData) * deviceBlockDurationMs
        additionalDeviceBlockIterations = numpy.lcm(d.NumberOfScans, blockSize) / samplingRate
        print(f"   first and every {additionalDeviceBlockIterations:g}'th GetData call must acquire an additional block from the device")
        print(f"   every {additionalDeviceBlockIterations}'th GetData dt:\t{additionalDeviceBlockDurationMs:.3f} ms ({math.ceil(deviceBlocksPerGetData)} device blocks @ {d.NumberOfScans} scans)")
        print(f"   intermediate GetData dt:\t{intermediateDurationMs:.3f} ms ({math.floor(deviceBlocksPerGetData)} device blocks @ {d.NumberOfScans} scans)")

    # acquisition (with time measurement)              
    sampleBlocks = []
    blockTimes = []
    acquiredSamples = 0
    nrBlocks = secondsToAcquire * samplingRate / blockSize
    avgBlockTime = 0
    start = timestamp()
    data=d.GetData(blockSize, retrieveData)
    stop = timestamp()

finally:
    icounter = d.IndexAfter('Counter')-1
    d.Close(); del d

# output elapsed time and length of samples
daqNettoDuration = elapsedMilliseconds(retrieveData.firstTime, retrieveData.lastTime)
daqTotalDuration = elapsedMilliseconds(start, stop)
print(f"Acquired {acquiredSamples} total samples in {daqTotalDuration:.3f} ms (including start/stop acquisition)\n"
      f"Acquired {acquiredSamples-blockSize} samples in {daqNettoDuration:.3f} ms (net)\n"
      f"Average sample block time: {avgBlockTime:.3f} ms")

# plot counter
if acquiredSamples > 0:
    scope = pygds.Scope(1/samplingRate, modal=True, ylabel='n', xlabel='t/s', title='Counter')
    scope(numpy.concatenate(sampleBlocks)[:, icounter:icounter+1])
    pyplot.show()
    
    pyplot.plot(blockTimes)
    pyplot.ylabel('Block Times')
    pyplot.show()
