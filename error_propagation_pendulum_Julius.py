import pandas as pd
import numpy as np
from pendul_waveform_data_Julius import p, ep
import matplotlib.pyplot as plt


li_raw = np.array([1.821, 1.822, 1.836, 1.821, 1.824])
ui_raw = np.array([.0005, .0005, .0008, .0003, .0005])
li = np.average(li_raw, weights=1/ui_raw**2)
ui = np.sqrt(1/np.sum(1/ui_raw**2))
lf_raw = np.array([1.836, 1.826, 1.839, 1.829, 1.826])
uf_raw = np.array([.0005, .0005, .0010, .0005, .0005])
lf = np.average(lf_raw, weights=1/uf_raw**2)
uf = np.sqrt(1/np.sum(1/uf_raw**2))
lr_raw = np.array([1.829, 1.836, 1.827, 1.832, 1.824])
ur_raw = np.array([.002, .003, .005, .005, .002])
lr = np.average(lr_raw, weights=1/ur_raw**2)
ur = np.sqrt(1/np.sum(1/ur_raw**2))

db_raw = np.array([.0607, .061, .0616, .0614, .0615])
ub_raw = np.array([.00005, .00005, .00005, .00005, .00005])
db = np.average(db_raw, weights=1/ub_raw**2)
ub = np.sqrt(1/np.sum(1/ub_raw**2))


plt.figure()
plt.hlines([li], xmin=-1-0.05, xmax=5-0.05, colors='C0')
plt.hlines([lf], xmin=-1, xmax=5, colors='C1')
plt.hlines([lr], xmin=-1+0.05, xmax=5+0.05, colors='C2')
plt.errorbar([-1-0.05, 5-0.05], [li, li], yerr=[ui, ui], fmt='.', color='C0',)
plt.errorbar(np.arange(5)-0.05, li_raw, yerr=ui_raw, fmt='.', color='C0', label='before')
plt.errorbar([-1, 5]          , [lf, lf], yerr=[uf, uf], fmt='.', color='C1')
plt.errorbar(np.arange(5)     , lf_raw, yerr=uf_raw, fmt='.', color='C1', label='after')
plt.errorbar([-1+0.05, 5+0.05], [lr, lr], yerr=[ur, ur], fmt='.', color='C2')
plt.errorbar(np.arange(5)+0.05, lr_raw, yerr=ur_raw, fmt='.', color='C2', label='ruler')
plt.legend()
# plt.show()


data_raw = pd.DataFrame(data={
    'L' : [lr+db/2], 
    'uL': [np.sqrt(ur**2 + (ub/2)**2)], 
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

s = np.abs(9.82 - data['g'])/data['ug']
# print(f'sigmas: {s[0]:.5f}')
data['sigmas'] = s
print(data.T)
plt.figure()
plt.plot([0], [9.82], '.')
plt.errorbar([0], data['g'], yerr=data['ug'], fmt='.')
plt.show()