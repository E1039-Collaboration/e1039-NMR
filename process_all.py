import NMRFastAna
import sys
import time
import os
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from scipy.optimize import curve_fit
from optparse import OptionParser

parser = OptionParser('Usage: %prog [options]')
parser.add_option('--log', type = 'string', dest = 'log', help = 'log file', default = '')
parser.add_option('--qcvpath', type = 'string', dest = 'qcvpath', help = 'path where qcv file is stored', default = './data')
parser.add_option('--qcvfile', type = 'string', dest = 'qcvfile', help = 'default qcurve file', default = 'QCV3563809315.csv')
parser.add_option('--mode', type = 'string', dest = 'mode', help = 'bkg subtraction method', default = 'spline')
parser.add_option('--path', type = 'string', dest = 'path', help = 'path to data', default = './data/')
parser.add_option('--type', type = 'string', dest = 'type', help = 'Type of data', default = 'TEQ')
parser.add_option('--freq', type = 'int', dest = 'freq', help = 'plot frequency', default = 100)
(options, args) = parser.parse_args()

dataDir = options.path
plotDir = './plot'
teData = [f for f in os.listdir(dataDir) if '.csv' in f and options.type in f]
logfile = open(options.log, 'w')

ana = NMRFastAna.NMRFastAna(options)
ana.freqWin = 0.25
ana.freqCenter = 212.97
ana.freqMin = 212.61
ana.freqMax = 213.39
ana.sampleSize = 0.002

tecalc = NMRFastAna.TEPolCalculator()

for idx,te in enumerate(teData):
    print te
    try:
        data = NMRFastAna.parseSweepFile(os.path.join(dataDir, te), True)
    except Exception, err:
        logfile.write('%s     %s\n' % (te, err))
        continue

    try:
        for i in range(len(data)):
            print '-------', i
            ana.setData(data[i])
            ana.setQcurve('auto')
            #ana.freqWin = 0.25
            #ana.qCurveAdjust()
            ana.qCurveSubtract()
            if i % options.freq == 0:
                ana.plot(plotDir, te.replace('.csv', '') + '_' + str(i))
            #print ana.signal.shortString()

            pol = 0.
            if options.type == 'TEQ':
                pol = tecalc.calcTEPol(data[i].HePress)

            logfile.write('%s   %s      %d    %.4f      %.4e      %.4e      %.4e     %.4e\n' % (te[8:13], te, i, ana.data.temp, ana.signal.peakX, ana.signal.peakY, ana.signal.integral, pol))
    except Exception, err:
        print te, err

    #dataAvg = NMRFastAna.parseSweepFile(os.path.join(dataDir, te))
    #ana.setData()

    logfile.flush()

logfile.close()
