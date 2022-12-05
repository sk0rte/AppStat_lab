from iminuit import Minuit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Pendul\Data\pendul_forsøg2.csv', header=14).drop(columns=['Channel 2 (V)']).rename(columns={'Time (s)': 't', 'Channel 1 (V)': 'A'})
# data = pd.read_csv('/Home\sk0rt3\Appstat projekt\AppStat_lab\Pendul\Data\pendul_forsøg2.csv', header=14).drop(columns=['Channel 2 (V)']).rename(columns={'Time (s)': 't', 'Channel 1 (V)': 'A'})

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

ey = np.ones_like(y)*(y_fit-y).std()

minuit = Minuit(chi2_owncalc, b=3.0, a=0.0)     # Own alternative

# Perform the actual fit:
minuit.migrad()
p = 1/minuit.values['a']
ep = minuit.errors['a']/minuit.values['a']**2

if __name__ == '__main__':
    print((y_fit-y).std())
    #Og så siger vi DNUR!
    print(f'{p:.5f} pm {ep:.5f}')

    # check if the found peaks match the data
    plt.figure()
    plt.plot('t', 'A', data=data)
    plt.plot(peaks, np.ones_like(peaks), '.')
    plt.show()

    # residual plot from fit
    plt.figure()
    plt.errorbar(x, y_fit-y, yerr=ey, fmt='.')
    plt.show()

    def gauss_pdf(x, mu, sigma):
        """Normalized Gaussian"""
        return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

    xx = np.linspace(-.01, 0.01, 1000)
    yy = gauss_pdf(xx, 0, ey[0])*len(x)*0.001

    plt.figure()
    plt.hist(y_fit-y, bins=np.arange(-.01, 0.0101, 0.001), histtype='step')
    plt.plot(xx, yy)
    plt.show()