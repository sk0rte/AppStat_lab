import pandas as pd
import numpy as np
from pendul_waveform_data_Julius import p, ep



data_raw = pd.DataFrame(data={
    'L' : [2], 
    'uL': [0.1], 
    'T' : [p], 
    'uT': [ep],
})
# print(data_raw)
data = pd.DataFrame()
data['L'] = [np.average(data_raw['L'], weights=1/data_raw['uL'])]
data['uL'] = np.sqrt(np.sum(data_raw['uL']**2))/len(data_raw)
data['T'] = [np.average(data_raw['T'], weights=1/data_raw['uT'])]
data['uT'] = np.sqrt(np.sum(data_raw['uT']**2))/len(data_raw)
data['g'] = data['L']*(2*np.pi/data['T'])**2
data['ug_L'] = 4*(np.pi/data['T'])**2*data['uL']
data['ug_T'] = 8*data['L']*np.pi**2/data['T']**3*data['uT']
data['ug'] = np.sqrt(data['ug_L']**2 + data['ug_T']**2)
print(data)