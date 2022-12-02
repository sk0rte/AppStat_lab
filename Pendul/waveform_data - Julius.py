from iminuit import Minuit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Pendul\Data\pendul_forsøg2.csv', header=14).drop(columns=['Channel 2 (V)']).rename(columns={'Time (s)': 't', 'Channel 1 (V)': 'A'})

idx = 1
peaks = []
in_peak = False
while idx < len(data):
    if data['A'][idx] > 1:
        if not in_peak:
            peaks.append(data['t'][idx])
        in_peak = True
    else:
        in_peak = False
    idx += 1
peaks.pop(0)
x = np.array(peaks)
y = np.arange(len(peaks))
ey = np.ones_like(y)

def fit_function(x, b, a):
    return b + a*x

# Alternatively, you can define Chi2 calculation:
def chi2_owncalc(b, a) :
    y_fit = fit_function(x, b, a)
    chi2 = np.sum(((y - y_fit) / ey)**2)
    return chi2
chi2_owncalc.errordef = 1.0    # Chi2 definition (for Minuit)

# Here we let Minuit know, what to minimise, how, and with what starting parameters:   
# minuit = Minuit(chi2_object, alpha0=3.0, alpha1=0.0)     # External Functions
minuit = Minuit(chi2_owncalc, b=3.0, a=0.0)     # Own alternative

# Perform the actual fit:
minuit.migrad()

y_fit: np.ndarray = fit_function(x, *minuit.values)

print((y_fit-y).std())
ey = ey*(y_fit-y).std()

minuit = Minuit(chi2_owncalc, b=3.0, a=0.0)     # Own alternative

# Perform the actual fit:
minuit.migrad()
p = 1/minuit.values['a']
ep = minuit.errors['a']/minuit.values['a']**2

print(f'{p:.5f} pm {ep:.5f}')


# plt.figure()
# plt.plot('t', 'A', data=data)
# plt.plot(peaks, np.ones_like(peaks), '.')
# # plt.plot(x, y_fit-y, '.')
# plt.show()