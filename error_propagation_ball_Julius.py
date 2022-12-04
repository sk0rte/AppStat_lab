import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def get_peak(data: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    t = data.loc[data['signal'].argmax(), 'time']
    d = 0.1
    c = 1.5
    piece = (data['signal']>c) & (data['time'] > t-d) & (data['time'] < t+d)
    data_p = data[piece]
    mi, ma = data_p['time'].min(), data_p['time'].max()
    return (data[~piece], [mi, ma])

def get_peaks(data: pd.DataFrame) -> list[float]:
    data, peak0 = get_peak(data)
    data, peak1 = get_peak(data)
    data, peak2 = get_peak(data)
    data, peak3 = get_peak(data)
    data, peak4 = get_peak(data)
    peaks = peak0 + peak1 + peak2 + peak3 + peak4
    peaks.sort()
    return peaks

norm_dir = pd.read_csv('Lab/testdata/data_NormDir_MedBall1.txt', delimiter='\t', header=None).rename(columns={0: 'time', 1: 'signal'})
data = pd.DataFrame(data={
    'a': [1],
    'ua': [.1],
    'theta': [1],
    'utheta': [.1],
    'D': [1],
    'uD': [.01],
    'd': [.5],
    'ud': [.1],
})


peaks = get_peaks(norm_dir)
print(peaks)

data['g'] = data['a']/np.sin(data['theta'])*(1+2/5*data['D']**2/(data['D']**2-data['d']**2))
data['ug_a'] = np.abs((1+2*data['D']**2/5/(data['D']**2-data['d']**2))/np.sin(data['theta'])*data['ua'])
data['ug_theta'] = np.abs(data['a']*(1+2*data['D']**2/5/(data['D']**2-data['d']**2))*data['utheta'])
data['ug_D'] = np.abs(data['a']*4/5*(data['D']/(data['D']**2-data['d']**2) - data['D']**3/(data['D']**2-data['d']**2)**2)/np.sin(data['theta'])*data['uD'])
data['ug_d'] = np.abs(4/5*data['a']*data['D']**2*data['d']/(data['D']**2-data['d']**2)**2/np.sin(data['theta'])*data['ud'])
data['ug'] = np.sqrt(data['ug_a']**2 + data['ug_theta']**2 + data['ug_D']**2 + data['ug_d']**2)
print(data)
