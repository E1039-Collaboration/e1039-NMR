import numpy as np
from scipy import interpolate
import sys
import NMRFastAna

# constants
## spline function for low pressure 0.0009 < P < 0.826
loPrange = (0.0009, 0.826)
loT = np.array([.650, .7, .75, .8, .85, .9, .95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25])
loP = np.array([1.101e-01, 2.923e-01, 6.893e-01, 1.475, 2.914, 5.380, 9.381, 15.58, 24.79, 38.02, 56.47, 81.52, 114.7])
loTfunc = interpolate.splrep(loT, loP, s = 0)

## poly function for medium pressure 0.826 < P < 37.82
mePrange = (0.826, 37.82)
mePara = np.array([1.392408, 0.527153, 0.166756], 0.050988, 0.026514, 0.001975, -.017976, 0.005409, 0.013259, 0.0])
meConst1 = 5.6
meConst2 = 2.9

## poly function for high pressure 37.82 < P < 1471.
hiPrange = (27.82, 1471.)
hiPara = np.array([3.146631, 1.357655, 0.413923, 0.091159, 0.016349, 0.001826, -.004325, -.004973, 0.0, 0.0])
hiConst1 = 10.3
hiConst2 = 1.9

## other consts
proton_g = 5.585694702
neutron_g = -3.82608545
deutron_g = 0.85741
nucleon_mag_moment = 5.05078353e-27
boltzmann = 1.38064852e-23
P_factor = nucleon_mag_moment*proton_g/boltzmann
D_factor = nucleon_mag_moment*deutron_g/boltzmann

def calcTEPol(P, B = 5., spin = 0.5, par = 'P'):
    T = calcT(P)
    if T < 0.:
        return 0.
    
    arg1 = (2.*spin + 1.)/spin/2.
    arg2 = 1./spin/2.
    fact = 0.
    if par == 'P':
        fact = P_factor*spin*B/T
    elif par == 'D':
        fact = D_factor*spin*B/T

    return arg1*np.cosh(arg1*fact)/np.sinh(arg1*fact) - arg2*np.cosh(arg2*fact)/np.sinh(arg2*fact)

def calcT(P):
    P = P*133.322
    if P >= loPrange[0] and P < loPrange[1]:
        return loTfunc.splev(P)
    elif P >= mePrange[0] and P < mePrange[1]:
        return np.polynomial.polynomial.polyval(np.log(P - meConst1)/meConst2, mePara)
    elif P >= hiPrange[0] and P < hiPrange[1]:
        return np.polynomial.polynomial.polyval(np.log(P - hiConst1)/hiConst2, hiPara)
    else:
        return -1.

def calcTEArea(teq, options):
    ana = NMRFastAna.NMRFastAna(options)

    # read te
    data = NMRFastAna.parseSweepFile(teq)
    ana.setData(data)
    ana.setQcurve('auto')
    ana.qCurveSubtract()

    if ana.signal.status == 2:
        return ana.signal.integral
    else:
        return 1.e9
    
def main(te, options):
    
    P = np.array([data.HePress for data in dataAll]).mean()

if __name__ == '__main__':
    parser = OptionParser('Usage: %prog [options]')
    parser.add_option('--teq', type = 'string', dest = 'teq', help = 'TEQ file name', default = '')
    parser.add_option('--qcvpath', type = 'string', dest = 'qcvpath', help = 'Path to data config', default = '')
    (options, args) = parser.parse_args()

    main(options)

    
    
    
    
    
    
