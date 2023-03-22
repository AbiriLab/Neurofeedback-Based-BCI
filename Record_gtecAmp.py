import pygds as g
import numpy as np
import pandas as pd
from time import perf_counter
d = g.GDS()

minf_s = sorted(d.GetSupportedSamplingRates()[0].items())[8] #8 refres to sampling rate=4800
d.SamplingRate, d.NumberOfScans = minf_s

for ch in d.Channels:
    ch.Acquire = True
d.SetConfiguration()
# scope = g.Scope(1/d.SamplingRate, title="Channels: %s", ylabel = u"U[μV]")

Matrixdata=[]
itteration=int(input('Enter the number of itterartions:')) #Itteration means the NumberOfScans
i=0
start_time = perf_counter()
while i<itteration:
    aa = d.GetData(d.SamplingRate)
    Matrixdata.extend(aa)
    i+=1
end_time = perf_counter() 
my_array = np.array(Matrixdata)
time_spent = end_time - start_time

print(time_spent)
print(d.SamplingRate)
pd.DataFrame(my_array).to_excel('my_matrix.xlsx')
