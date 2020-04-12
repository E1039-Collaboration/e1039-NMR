import numpy as np
import sys
from optparse import OptionParser
import NMRFastAna

def main(options):

    # initialize the analyzer
    ana = NMRFastAna.NMRFastAna(options)
    pol = NMRFastAna.TEPolCalculator()

    # read the TEQ file
    tedata = NMRFastAna.parseSweepFile(options.teqfile, True)

    # extract all runs
    p_raw = []
    i_raw = []
    for data in tedata:
        try:
            ana.setData(data)
            ana.setQcurve('auto')
            ana.qCurveSubtract()
            p = pol.calcTEPol(data.HePress)

            p_raw.append(p)
            i_raw.append(ana.signal.integral)
        except:
            continue
    p_raw = np.array(p_raw)
    i_raw = np.array(i_raw)

    # reject outliers
    p_mean = p_raw.mean()
    p_sigma = p_raw.std()
    i_mean = i_raw.mean()
    i_sigma = i_raw.std()
    
    p_final = []
    i_final = []
    for i in range(len(p_raw)):
        if abs(p_raw[i] - p_mean) < 3.*p_sigma and abs(i_raw[i] - i_mean) < 3.*i_sigma:
            p_final.append(p_raw[i])
            i_final.append(i_raw[i])

    #print len(p_final), np.array(p_final).mean(), np.array(i_final).mean()
    print np.array(p_final).mean()/np.array(i_final).mean()*100

if __name__ == '__main__':
    parser = OptionParser('Usage: %prog [options]')
    # requested in NMRFastAna
    parser.add_option('--qcvpath', type = 'string', dest = 'qcvpath', help = 'path where qcv file is stored', default = './data')
    parser.add_option('--qcvfile', type = 'string', dest = 'qcvfile', help = 'default qcurve file', default = 'QCV3563809315.csv')
    parser.add_option('--mode', type = 'string', dest = 'mode', help = 'bkg subtraction method', default = 'spline')

    # for this job only
    parser.add_option('--teqfile', type = 'string', dest = 'teqfile', help = 'TEQ file', default = '')
    (options, args) = parser.parse_args()
    
    main(options)
