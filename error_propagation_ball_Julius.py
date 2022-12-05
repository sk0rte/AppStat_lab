import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from iminuit import Minuit

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

norm_dir = pd.read_csv('Ball/Data/forsøg1_bigball_left.csv', delimiter=',', header=13).rename(columns={'Time (s)': 'time', 'Channel 1 (V)': 'signal'})
peaks = get_peaks(norm_dir)
d = 0.05
p = np.linspace(0, 1.9, 5)#[.5, .75, 1, 1.25, 1.5]
pos = np.kron(p, [1,1])+np.kron([1,1,1,1,1], [-d/2, d/2])

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

x = np.array(peaks)
y = np.array(pos)
ey = 0.0025
def fit(x, a, b, c):
    return a*x**2 + b*x + c

def chi2(a, b, c):
    y_fit = fit(x, a, b, c)
    return np.sum(((y - y_fit) / ey)**2)
chi2.errordef = 1.0

minuit = Minuit(chi2, a=1, b=0, c=0)
minuit.migrad()
print(minuit.values)
print(minuit.errors)
print(f'reduced chi2: {minuit.fval / 7 :.2f}')

xx = np.linspace(x[0], x[-1], 1000)
yy = fit(xx, *minuit.values)
plt.figure()
plt.plot(xx, yy)
plt.plot(x, y, '.')
plt.show()

data['g'] = data['a']/np.sin(data['theta'])*(1+2/5*data['D']**2/(data['D']**2-data['d']**2))
data['ug_a'] = np.abs((1+2*data['D']**2/5/(data['D']**2-data['d']**2))/np.sin(data['theta'])*data['ua'])
data['ug_theta'] = np.abs(data['a']*(1+2*data['D']**2/5/(data['D']**2-data['d']**2))*data['utheta'])
data['ug_D'] = np.abs(data['a']*4/5*(data['D']/(data['D']**2-data['d']**2) - data['D']**3/(data['D']**2-data['d']**2)**2)/np.sin(data['theta'])*data['uD'])
data['ug_d'] = np.abs(4/5*data['a']*data['D']**2*data['d']/(data['D']**2-data['d']**2)**2/np.sin(data['theta'])*data['ud'])
data['ug'] = np.sqrt(data['ug_a']**2 + data['ug_theta']**2 + data['ug_D']**2 + data['ug_d']**2)
print(data)
