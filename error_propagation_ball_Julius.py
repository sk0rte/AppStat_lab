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

result = pd.DataFrame()
for j in ['bigball', 'smallball']:
    for k in ['left', 'right']:
        data_list = []
        a_all = []
        ua_all = []
        for i in range(1, 6, 1):
            print(f'Forsøg {i} with the {j} going in the {k} direction:')

            # timing
            norm_dir = pd.read_csv(f'Ball/Data/forsøg{i}_{j}_{k}.csv', delimiter=',', header=13).rename(columns={'Time (s)': 'time', 'Channel 1 (V)': 'signal'})
            peaks = get_peaks(norm_dir)

            # Angle
            if k == 'left':
                theta_data = np.array([77, 78, 77, 77, 75.5, 75.5, 76, 74, 75, 76.5])
            else:
                theta_data = np.array([74, 74, 76, 74.5, 74, 76, 77.5, 75, 76.5, 76.5])
            theta = (90-theta_data.mean()) * np.pi / 180
            utheta = theta_data.std() / np.sqrt(10) * np.pi / 180

            # ball radius
            if j == 'bigball':
                D_data = np.array([15.7, 15.8, 15.9, 15.9, 15.9]) / 1000
            else:
                D_data = np.array([11.6, 11.9, 11.9, 11.9, 12.0]) / 1000
            D = D_data.mean()
            uD = 0.0001 / np.sqrt(5)

            # rail distance
            d_data = np.array([5.9, 6.0, 5.9, 5.8, 5.7]) / 1000
            d = d_data.mean()
            ud = 0.00005
            
            pos_data = np.array([[130, 273, 416, 558, 700],
                                 [129, 271, 414, 556, 700],
                                 [129, 271, 415, 557, 700],
                                 [135, 276, 420, 562, 705],
                                 [130, 271, 416, 558, 701]]) / 1000
            pos = np.kron(pos_data.mean(axis=0), [1,1])+np.kron([1,1,1,1,1], [-D/2, D/2])
            # print(pos)

            t = np.array(peaks)
            x = np.array(pos)
            ex = 0.004/np.sqrt(5)
            def fit(x, a, b, c):
                return a*x**2/2 + b*x + c

            def chi2(a, b, c):
                y_fit = fit(t, a, b, c)
                return np.sum(((x - y_fit) / ex)**2)
            chi2.errordef = 1.0

            minuit = Minuit(chi2, a=1, b=0, c=0)
            minuit.migrad()
            # print(minuit.values)
            # print(minuit.errors)
            # print(f'reduced chi2: {minuit.fval / 7 :.2f}')
            a_all.append(minuit.values['a'])
            ua_all.append(minuit.errors['a'])

            xx = np.linspace(t[0], t[-1], 1000)
            yy = fit(xx, *minuit.values)
            plt.figure()
            plt.plot(xx, yy)
            plt.plot(t, x, '.')
            # plt.show()

        data = pd.DataFrame(data={
            'a': [np.average(a_all, weights=ua_all)],
            'ua': [np.sqrt(1/np.sum(1/np.array(ua_all)**2))],
            'theta': [theta],
            'utheta': [utheta],
            'D': [D],
            'uD': [uD],
            'd': [d],
            'ud': [ud],
        })


        data['g'] = data['a']/np.sin(data['theta'])*(1+2/5*data['D']**2/(data['D']**2-data['d']**2))
        data['ug_a'] = np.abs((1+2*data['D']**2/5/(data['D']**2-data['d']**2))/np.sin(data['theta'])*data['ua'])
        data['ug_theta'] = np.abs(data['a']*(1+2*data['D']**2/5/(data['D']**2-data['d']**2))*data['utheta'])
        data['ug_D'] = np.abs(data['a']*4/5*(data['D']/(data['D']**2-data['d']**2) - data['D']**3/(data['D']**2-data['d']**2)**2)/np.sin(data['theta'])*data['uD'])
        data['ug_d'] = np.abs(4/5*data['a']*data['D']**2*data['d']/(data['D']**2-data['d']**2)**2/np.sin(data['theta'])*data['ud'])
        data['ug'] = np.sqrt(data['ug_a']**2 + data['ug_theta']**2 + data['ug_D']**2 + data['ug_d']**2)

        result[f"{j.removesuffix('ball')}{k.capitalize()}"] = data.T
print(result)
