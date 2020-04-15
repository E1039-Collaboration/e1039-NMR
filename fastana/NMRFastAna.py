import sys
import os
import socket
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import select
from scipy import interpolate
from scipy.optimize import curve_fit
from iminuit import Minuit
from datetime import datetime
from optparse import OptionParser
import logging

class logger:
    """Configure a simple logger for both console and file output"""
    log_console = logging.getLogger('console')
    log_fileout = logging.getLogger('fileout')
    enableFileout = False

    @staticmethod
    def init(options):
        log_format = '%(asctime)s - %(levelname)s: %(message)s'
        log_handler_console = logging.StreamHandler()
        log_handler_console.setFormatter(logging.Formatter(log_format))
        logger.log_console.setLevel(options.loglevel.upper())
        logger.log_console.addHandler(log_handler_console)

        logging.enableFileout = False
        if options.logfile != '':
            log_handler_fileout = logging.FileHandler(options.logfile)
            log_handler_fileout.setFormatter(logging.Formatter(log_format))     
            logger.log_fileout.setLevel(options.loglevel)
            logger.log_fileout.addHandler(log_handler_fileout)

            logger.enableFileout = True

    @staticmethod
    def debug(msg):
        logger.printlog('debug', msg)
    
    @staticmethod
    def info(msg):
        logger.printlog('info', msg)

    @staticmethod
    def warning(msg):
        logger.printlog('warning', msg)
    
    @staticmethod
    def error(msg):
        logger.printlog('error', msg)

    @staticmethod
    def critical(msg):
        """Critical is used for exception handling, a bit more infomation is added"""
        _, exc_obj, tb = sys.exc_info()
        lineno = tb.tb_lineno
        logger.printlog('critical', '%s - line %d - %s' % (msg, lineno, str(exc_obj)))

    @staticmethod
    def printlog(level, msg):
        getattr(logger.log_console, level)(msg)
        if logger.enableFileout:
            getattr(logger.log_fileout, level)(msg)

class SweepData:
    """Container of sweep data"""
    #gain_value = [1., 20., 200.]
    gain_value = [1., 1., 1.]

    def __init__(self, inputStr = '', qfile = '', ts = 0, center = 213., freqMin = 212.6, freqMax = 213.4, nSteps = 500, gain = 1, polarity = 1., amps = []):
        """parse the input string and fill the header+data info"""
        self.polarity = polarity
        if inputStr == '':  # if nothing is provided, put together a fake header
            fakeHeader = ['9999.9999' for i in range(35)]
            fakeHeader[0] = 'Data from UVa q-meter'
            fakeHeader[1] = qfile
            fakeHeader[3] = 'teq.csv'
            fakeHeader[12] = '%f' % ((freqMin + freqMax)/2.)
            fakeHeader[13] = '%f' % ((freqMax - freqMin)/nSteps)
            fakeHeader[14] = '%f' % (nSteps + 1)   # Attention: compensate for the last point
            fakeHeader[24] = '%f' % gain
            fakeHeader[27] = '0.0000'
            fakeHeader[16] = '22.0000'
            fakeHeader[15] = '%.4f' % ts
            fakeHeader[33] = '1.0000'
            fakeHeader[34] = '1.0000'
            fakeHeader[10] = '%s.0000' % str(long(ts) + long(2082844800))[:5]
            fakeHeader[11] = '%s.0000' % str(long(ts) + long(2082844800))[5:]

            inputStr = '\n'.join(fakeHeader+amps+[amps[-1]])  # Attention, compensate for the last point

        info = inputStr.strip().split('\n')
        assert len(info) > 36, 'input string too short'

        # direct data
        self.header     = info[:35]
        self.QCurveFile = info[1].strip()
        self.TEFile     = info[3].strip()
        self.centerFreq = float(info[12])
        self.stepSize   = float(info[13])
        self.nSteps     = int(float(info[14]))
        self.gain       = int(float(info[24]))
        self.logScale   = int(float(info[27])) == 1
        self.temp       = float(info[16])
        self.sweepID    = int(float(info[15]))
        self.HeTemp     = float(info[33])
        self.HePress    = float(info[34])
        self.timeH      = int(float(info[10]))
        self.timeL      = int(float(info[11]))
        try:
            self.signalL    = float(info[8])
            self.signalH    = float(info[9])
        except:
            self.signalL    = 212.75
            self.signalH    = 213.25

        # indirect ones
        self.minFreq = self.centerFreq - (self.nSteps - 1.)/2.*self.stepSize
        self.maxFreq = self.centerFreq + (self.nSteps - 1.)/2.*self.stepSize

        # range of valid data, default to be the entire data range
        self.validL = self.minFreq
        self.validH = self.maxFreq

        # x and y data
        self.freq = np.linspace(self.minFreq, self.maxFreq, num = self.nSteps, endpoint = True)
        self.amp  = np.array([float(line)/SweepData.gain_value[self.gain] for line in info[35:]])

        # these are just place holders for future use
        self.peakIdx = self.getPeakIdx()
        self.peakX = self.freq[self.peakIdx]
        self.peakY = self.amp[self.peakIdx]*SweepData.gain_value[self.gain]
        self.HML   = self.peakX - 0.06   # if we consider the 0.2MHz is 3sigma
        self.HMR   = self.peakX + 0.06

        # basic sanity check
        assert self.amp.size == self.nSteps, 'step size in header does not match data'
        assert self.nSteps > 50, 'number of steps too small'
        assert self.gain <= 2, 'gain selection wrong'

        # spline representation
        self.func = None

        # integral of the curve
        self.integral = 0.

        # number of average events
        self.evtCounts = 1

        # flag of bkg-subtraction status, 0 for fail, 1 for success
        self.status = 0

    def update(self, avgwin = 3):
        """update the peak info using the most recent data"""
        self.peakIdx = self.getPeakIdx()
        peakSlice = np.arange(self.peakIdx-avgwin, self.peakIdx+avgwin+1, 1)

        self.peakX = (self.freq[peakSlice]*self.amp[peakSlice]).sum()/self.amp[peakSlice].sum()
        self.peakY = np.interp(self.peakX, self.freq, self.amp)*SweepData.gain_value[self.gain]

        uspline = interpolate.UnivariateSpline(self.freq, self.amp - 0.5*self.peakY, s = 0)
        self.HML, self.HMR = uspline.roots()[:2]
    
    def getPeakIdx(self):
        return self.getPeakFunc()(self.amp)
        
    def getPeakFunc(self):
        if self.polarity > 0.:
            return np.argmax
        else:
            return np.argmin
    
    def getArea(self, fmin, fmax):
        signalSlice = [i for i in range(self.freq.size) if self.freq[i] > fmin and self.freq[i] < fmax]
        return self.amp[signalSlice].sum()*self.stepSize

    def shortString(self):
        """return the data plus a short header: 1. number of points; 2. starting frequency; 3. frequency step; 4. the integral of the curve"""
        s = ''
        h = '%d\n%.4f\n%.4f\n%s\n' % (self.status, self.peakX, self.peakY, str(self.integral*SweepData.gain_value[self.gain]))
        for item in self.amp:
            s = s + '%.4f\n' % (item*SweepData.gain_value[self.gain])
        return h, s.strip()

    def longString(self):
        """return the long string following the same format as input"""
        l = ''
        for item in self.header:
            l = l + item + '\n'
        for item in self.amp:
            l = l + '%f\n' % (item*SweepData.gain_value[self.gain])
        return l

    def interpolate(self):
        """create spline interpolation for future use"""
        self.func = interpolate.splrep(self.freq, self.amp, s = 0)
    
    @staticmethod
    def parseSweepFile(filename, average = 9999):
        """parse an entire input file. if it's not an AVE file produced by LabView, read the entire file and make average"""
        assert os.path.exists(filename) or os.path.exists(filename.replace('AVE', '')), 'File %s does not exist.' % filename
        assert average > 0, 'Average window cannot be smaller than 1, %d was provided.' % average

        if 'AVE' in filename and os.path.exists(filename):
            return SweepData('\n'.join([line.strip() for line in open(filename)]))
        else: # if AVE file does not exist, try the original one and calculate the average myself ...
            filename = filename.replace('AVE', '')
            contents = [line.strip() for line in open(filename)]

            index = 0
            rawSweeps = []
            while index < len(contents):
                nSteps = int(float(contents[index + 14]))
                rawSweeps.append(SweepData('\n'.join(contents[index:index + 35 + nSteps])))
                index = index + 35 + nSteps
            
            sweeps = []
            for i, s in enumerate(rawSweeps[1:]):
                if (i % average) == 0:
                    sweep = copy.deepcopy(s)
                    nAvg = 1.
                    continue
                
                sweep.amp = sweep.amp + s.amp
                sweep.timeH = sweep.timeH + s.timeH
                sweep.timeL = sweep.timeL + s.timeL
                nAvg = nAvg + 1.
                
                if (i % average) == (average-1) or i == len(rawSweeps)-2:
                    sweep.amp = sweep.amp/nAvg
                    sweep.timeH = int(sweep.timeH/nAvg)
                    sweep.timeL = int(sweep.timeL/nAvg)
                    sweep.evtCounts = int(nAvg + 0.5)
                    sweeps.append(sweep)
                    continue

            return sweeps

class TEPolCalculator:
    def __init__(self):
        # constants
        ## spline function for low pressure 0.0009 < P < 0.826
        self.loPrange = (0.0009, 0.826)
        self.loT = np.array([.650, .7, .75, .8, .85, .9, .95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25])
        self.loP = np.array([1.101e-01, 2.923e-01, 6.893e-01, 1.475, 2.914, 5.380, 9.381, 15.58, 24.79, 38.02, 56.47, 81.52, 114.7])
        self.loTfunc = interpolate.splrep(self.loP, self.loT, s = 0)

        ## poly function for medium pressure 0.826 < P < 37.82
        self.mePrange = (0.826, 37.82)
        self.mePara = np.array([1.392408, 0.527153, 0.166756, 0.050988, 0.026514, 0.001975, -.017976, 0.005409, 0.013259, 0.0])
        self.meConst1 = 5.6
        self.meConst2 = 2.9

        ## poly function for high pressure 37.82 < P < 1471.
        self.hiPrange = (27.82, 1471.)
        self.hiPara = np.array([3.146631, 1.357655, 0.413923, 0.091159, 0.016349, 0.001826, -.004325, -.004973, 0.0, 0.0])
        self.hiConst1 = 10.3
        self.hiConst2 = 1.9

        ## other consts
        self.proton_g = 5.585694702
        self.neutron_g = -3.82608545
        self.deutron_g = 0.85741
        self.nucleon_mag_moment = 5.05078353e-27
        self.boltzmann = 1.38064852e-23
        self.P_factor = self.nucleon_mag_moment*self.proton_g/self.boltzmann
        self.D_factor = self.nucleon_mag_moment*self.deutron_g/self.boltzmann

    def calcTEPol(self, P, B = 5., spin = 0.5, par = 'P'):
        T = self.calcT(P)
        if T < 0.:
            return 0.

        arg1 = (2.*spin + 1.)/spin/2.
        arg2 = 1./spin/2.
        fact = 0.
        if par == 'P':
            fact = self.P_factor*spin*B/T
        elif par == 'D':
            fact = self.D_factor*spin*B/T

        return arg1*np.cosh(arg1*fact)/np.sinh(arg1*fact) - arg2*np.cosh(arg2*fact)/np.sinh(arg2*fact)

    def calcT(self, P):
        if P >= self.loPrange[0] and P < self.loPrange[1]:
            return float(interpolate.splev(P*133.322, self.loTfunc, der = 0))
        elif P >= self.mePrange[0] and P < self.mePrange[1]:
            return np.polynomial.polynomial.polyval((np.log(P*133.322) - self.meConst1)/self.meConst2, self.mePara)
        elif P >= self.hiPrange[0] and P < self.hiPrange[1]:
            return np.polynomial.polynomial.polyval((np.log(P*133.322) - self.hiConst1)/self.hiConst2, self.hiPara)
        else:
            return -1.

class NMRFastAna:

    def __init__(self, options = None):
        self.freqCenter = 213.
        self.freqWin = 0.25
        self.freqMin = 212.65  # real frequency range of the input data
        self.freqMax = 213.35
        self.freqValidMin = self.freqMin  # valid range of the input data
        self.freqValidMax = self.freqMax 
        self.freqAdjMin = self.freqMin  # the range used the background subtraction after optional q-curve shift
        self.freqAdjMax = self.freqMax
        self.sampleRate = 1

        self.xOffsetMin = -0.5
        self.xOffsetMax = 0.5
        self.yOffsetMin = -1.
        self.yOffsetMax = 1.
        self.xOffset    = 0.
        self.yOffset    = 0.
        self.scale      = 1.

        # data is for raw data, like TEQ or POL
        # ref is for background, i.e. QCV
        self.data = None
        self.ref = None

        # file name and path of the QCV file
        self.refFile = ''   # in principle each SweepData header should contain the QCV file to use
        self.refPath = ''   
        self.refDefaultFile = ''  # if refFile is not provided or available, use the default instead
        if options is not None:
            self.refPath = options.qcvpath
            self.refDefaultFile = options.qcvfile

        # q curve subtracted data, X and Y and function
        self.subtractedX = None
        self.subtractedY = None
        self.subtractedF = None

        # subset of q qurve subtracted data, only in the pre-defined sideband region
        self.sidebandX = None
        self.sideBandY = None
        self.sidebandF = None   # spline function for sideband
        self.sidebandP = None   # polymonial function for sideband
        self.sidebandC = None   # Chebyshev function for sideband

        # bkg-subtraction mode
        self.mode = 'spline'
        if options is not None:
            self.mode = options.mode

        # pure signal after furture background subtraction with spline, in the form of SweepData
        self.signal = None

        # container of the results so far
        self.results = []

        # flag about whether Q curve subtraction is needed
        self.qcurveless = False
        if options is not None:
            self.qcurveless = options.qcvless

        self.minimizer = None

    def runningAvg(self, window = 10):
        id_start = max(0, len(self.results) - window)

        avgSweep = copy.deepcopy(self.signal)
        for s in self.results[id_start:len(self.results)-1]:
            avgSweep.amp = avgSweep.amp + s.amp

        avgSweep.amp = avgSweep.amp/(len(self.results) - id_start)
        return avgSweep

    def setQcurve(self, qfile):
        """read the Q curve file, if input file set to auto, it will read the q curve file specified in data header instead. return True if succeeded"""
        if qfile == 'auto':
            qfile = self.data.QCurveFile

        # if the QCurveLess is specified earlier, just ignore
        if self.qcurveless:
            return

        # if the Q curve file is ignore, run in qcurveless mode
        if 'ignore' in qfile:
            self.qcurveless = True
            return

        # check if either AVE file or original file exists
        qfileAve = qfile.replace('AVE', '')
        qfileAbs = os.path.join(self.refPath, qfile)
        qfileAbsAve = qfileAbs.replace('AVE', '')
        qfileAbsDefault = os.path.join(self.refPath, self.refDefaultFile)
        assert (os.path.exists(qfileAbs) or os.path.exists(qfileAbsAve) or os.path.exists(qfileAbsDefault)), 'Q curve input file %s (or %s) does not exist.' % (qfileAbs, qfileAbsDefault)

        if os.path.exists(qfileAbs):
            if self.refFile != qfile:
                self.refFile = qfile
                self.ref = SweepData.parseSweepFile(qfileAbs)[0]
                self.ref.interpolate()
        elif os.path.exists(qfileAbsAve):
            if self.refFile != qfileAve:
                self.refFile = qfileAve
                self.ref = SweepData.parseSweepFile(qfileAbsAve)[0]
                self.ref.interpolate()
        elif os.path.exists(qfileAbsDefault):
            if self.refFile != self.refDefaultFile:
                self.refFile = self.refDefaultFile
                self.ref = SweepData.parseSweepFile(qfileAbsDefault)[0]
                self.ref.interpolate()

    def setData(self, dataIn):
        """set the data input"""
        self.data = dataIn
        self.data.interpolate()

        self.xOffset = 0.
        self.yOffset = 0.

        self.freqCenter = 0.5*(self.data.signalL + self.data.signalH)
        self.freqWin    = 0.5*(self.data.signalH - self.data.signalL)

        self.freqMin = self.data.freq[0]
        self.freqMax = self.data.freq[-1]
        self.freqValidMin = self.data.validL
        self.freqValidMax = self.data.validH

        # check if it's from a new run
        if len(self.results) > 0 and self.data.sweepID < self.results[-1].sweepID:
            self.clearCache()

    def clearCache(self):
        self.results = []

    def range(self):
        """adjust the range according to the current offset parameter"""
        # find the real minimum and maximum given the offset
        return (max(self.freqValidMin+self.xOffset, self.freqValidMin), min(self.freqValidMax+self.xOffset, self.freqValidMax))

    def chisq(self, x, y):
        """definition of the chi^2 function for Q curve adjustment"""
        self.xOffset = x
        self.yOffset = y
        self.freqAdjMin, self.freqAdjMax = self.range()

        freq = np.array([f for f in self.ref.freq[range(0, self.ref.freq.size, self.sampleRate)] if abs(f - self.freqCenter) > self.freqWin and f > self.freqAdjMin and f < self.freqAdjMax])
        chi2 = ((interpolate.splev(freq, self.data.func, der = 0) - (interpolate.splev(freq - x, self.ref.func, der = 0) + y))**2).sum()/freq.size

        return chi2

    def qCurveAdjust(self):
        """use sideband region for the Q curve adjustment"""
        if self.qcurveless:
            return

        self.minimizer = Minuit(self.chisq, x = self.xOffset, error_x = 0.1, limit_x = (self.xOffsetMin, self.xOffsetMax), y = self.yOffset, error_y = 0.1, limit_y = (self.yOffsetMin, self.yOffsetMax), errordef = 1, print_level = 0)
        self.minimizer.migrad()
        #self.minimizer.print_param()
        #print self.minimizer.get_fmin()

        if self.minimizer.get_fmin().edm < 5.E-6:
            self.xOffset = self.minimizer.values['x']
            self.yOffset = self.minimizer.values['y']
        else:
            self.xOffset = 0.
            self.yOffset = 0.

    def smoothArray(self, xx):
        """Implementation of HBOOK 353QH algorithm, ref: Proc.of the 1974 CERN School of Computing, Norway, 11-24 August, 1974. page 293"""
        # only applicable to size > 3 array
        nn = xx.size
        if nn < 3:
            return

        # temp arraies
        yy = np.zeros(xx.size)
        zz = xx.copy()
        rr = np.zeros(xx.size)

        # 353QH twice
        for i in range(2):
            # 3, 5, 3 running medians
            for j in range(3):
                np.copyto(yy, zz)

                # 3 or 5 median
                medianWin = 1
                ifirst = 1
                ilast = nn - 1
                if j == 1:
                    medianWin = 2
                    ifirst = 2
                    ilast = nn - 2

                # the middle range
                for k in range(ifirst, ilast):
                    zz[k] = np.median(yy[range(k-medianWin, k+medianWin+1)])

                # the edge case
                if j == 0: # first 3-median
                    zz[0] = np.median([zz[0], zz[1], 3.*zz[1] - 2.*zz[2]])
                    zz[nn-1] = np.median([zz[nn-2], zz[nn-1], 3.*zz[nn-2] - 2.*zz[nn-3]])
                elif j == 1:
                    zz[1] = np.median([yy[0], yy[1], yy[2]])
                    zz[nn-2] = np.median([yy[nn-3], yy[nn-2], yy[nn-1]])

            # update the temp array
            np.copyto(yy, zz)

            # quadratic interpolation on flat segments
            for j in range(2, nn-2):
                if zz[j-1] != zz[j] or zz[j] != zz[j+1]:
                    continue

                hh0 = zz[j-2] - zz[j]
                hh1 = zz[j+2] - zz[j]
                if hh0*hh1 <= 0:
                    continue

                jk = 1
                if abs(hh1) > abs(hh0):
                    jk = -1

                yy[j] = -0.5*zz[j-2*jk] + zz[j]/0.75 + zz[j+2*jk]/6.
                yy[j+jk] = 0.5*(zz[j+2*jk] - zz[j-2*jk]) + zz[j]

            # running means
            for j in range(1, nn-1):
                zz[j] = 0.25*yy[j-1] + 0.5*yy[j] + 0.25*yy[j+1]
            zz[0] = yy[0]
            zz[nn-1] = yy[nn-1]

            # if it's the first run
            if i == 0:
                np.copyto(rr, zz)
                zz = xx - zz

        xx[:] = zz + rr

    def fitSideband(self, mode = None):
        if mode == None:
            mode = self.mode

        if mode == 'spline':
            self.sidebandF = interpolate.splrep(self.sidebandX, self.sidebandY, s = 1, k = 3)
        elif mode == 'poly3':
            self.sidebandP = np.poly1d(np.polyfit(self.sidebandX, self.sidebandY, 3))
        elif mode == 'cheby3':
            self.sidebandC = np.polynomial.Chebyshev.fit(self.sidebandX, self.sidebandY, 3)
    
    def bkgY(self, bkgX, mode = None):
        if mode == None:
            mode = self.mode

        if mode == 'spline':
            return interpolate.splev(bkgX, self.sidebandF, der = 0)
        elif mode == 'poly3':
            return self.sidebandP(bkgX)
        elif mode == 'cheby3':
            return self.sidebandC(bkgX)
    
    def bkgChi2(self, mode = None):
        if mode == None:
            mode = self.mode
        
        return ((self.sidebandY - self.bkgY(self.sidebandX, mode=mode))**2).sum()/self.sidebandX.size
       
    def qCurveSubtract(self):
        """with the adjusted Q curve info, make the Q curve subtraction and then the sideband spline subtraction"""
        self.freqAdjMin, self.freqAdjMax = self.range()

        # subtract Q curve from the data
        subtractedSlice = np.array([i for i in range(self.data.freq.size) if self.data.freq[i] > self.freqAdjMin and self.data.freq[i] < self.freqAdjMax])
        self.subtractedX = self.data.freq[subtractedSlice]
        if self.qcurveless:
            self.subtractedY = self.data.amp[subtractedSlice]
        else:
            self.subtractedY = self.data.amp[subtractedSlice] - interpolate.splev(self.subtractedX - self.xOffset, self.ref.func, der = 0) - self.yOffset
        #self.subtractedF = interpolate.splrep(self.subtractedX, self.subtractedY, s = 0)

        # smooth the left and right sideband
        adjustedCenter = self.freqCenter  #self.subtractedX[np.argmax(self.subtractedY)]
        RsidebandSliceL = np.array([i for i in range(0, self.subtractedX.size, self.sampleRate) if adjustedCenter - self.subtractedX[i] > self.freqWin])
        RsidebandSliceR = np.array([i for i in range(0, self.subtractedX.size, self.sampleRate) if self.subtractedX[i] - adjustedCenter > self.freqWin])

        #sidebandMeanL = self.subtractedY[RsidebandSliceL].mean()
        #sidebandSigmaL = self.subtractedY[RsidebandSliceL].std()
        #sidebandMeanR = self.subtractedY[RsidebandSliceR].mean()
        #sidebandSigmaR = self.subtractedY[RsidebandSliceR].std()
        #sidebandSliceL = np.array([i for i in range(0, self.subtractedX.size, sampleRate) if adjustedCenter - self.subtractedX[i] > self.freqWin and abs(self.subtractedY[i] - sidebandMeanL) < 2.*sidebandSigmaL])
        #sidebandSliceR = np.array([i for i in range(0, self.subtractedX.size, sampleRate) if self.subtractedX[i] - adjustedCenter > self.freqWin and abs(self.subtractedY[i] - sidebandMeanR) < 2.*sidebandSigmaR])
        sidebandSlice = np.concatenate((RsidebandSliceL, RsidebandSliceR))

        self.sidebandX = self.subtractedX[sidebandSlice]
        self.sidebandY = self.subtractedY[sidebandSlice]

        # 353QH smoothing
        self.smoothArray(self.sidebandY[:RsidebandSliceL.size])
        self.smoothArray(self.sidebandY[RsidebandSliceL.size+1:])

        # subtract either interpolated/fitted sideband from Q-curve-subtracted data
        self.signal = copy.deepcopy(self.data)
        self.fitSideband()
        signalY = self.subtractedY - self.bkgY(self.subtractedX)

        # add zeros on left/right if necessary to the data to match the input data length
        paddingL = np.array([f for f in self.data.freq if f < self.subtractedX.min()])
        paddingH = np.array([f for f in self.data.freq if f > self.subtractedX.max()])
        self.signal.freq = np.concatenate((paddingL, self.subtractedX, paddingH))
        self.signal.amp  = np.concatenate((np.full(paddingL.size, 0.), signalY, np.full(paddingH.size, 0.)))
        self.signal.update()

        # only integrate the curve inside the signal window
        self.signal.integral = self.signal.getArea(adjustedCenter - self.freqWin, adjustedCenter + self.freqWin)

        # check the status of the bkg-subtraction
        self.signal.status = 0
        # sidebandSlopeL = (self.subtractedY[RsidebandSliceL[-1]] - self.subtractedY[RsidebandSliceL[0]])/(self.subtractedX[RsidebandSliceL[-1]] - self.subtractedX[RsidebandSliceL[0]])
        # sidebandSlopeR = (self.subtractedY[RsidebandSliceR[-1]] - self.subtractedY[RsidebandSliceR[0]])/(self.subtractedX[RsidebandSliceR[-1]] - self.subtractedX[RsidebandSliceR[0]])
        # if abs(sidebandSlopeL - sidebandSlopeR)/(abs(sidebandSlopeL) + abs(sidebandSlopeR)) < 1.:
        #     self.signal.status = 2
        # else:
        #     self.signal.status = 1

        self.results.append(self.signal)

    def plot(self, path, prefix):
        saveprefix = os.path.join(path, prefix)
        freqaxis = np.linspace(self.freqAdjMin, self.freqAdjMax, num = 1000, endpoint = True)
        freqaxis_short = np.linspace(self.signal.HML, self.signal.HMR, num = 100, endpoint = True)

        # raw data + raw q curve
        plt.figure(0)
        plt.plot(self.data.freq, self.data.amp, '.', color = 'red')
        if not self.qcurveless:
            plt.plot(self.ref.freq, self.ref.amp, '.', color = 'blue')
        plt.savefig(saveprefix + '_raw.png')
        plt.close()

        # bkg subtraction overlay -- focus on bkg
        plt.figure(1)
        plt.plot(self.subtractedX, self.subtractedY, '.')
        plt.plot(self.sidebandX, self.sidebandY, '.', color = 'red')
        if self.mode == 'spline':
            plt.plot(freqaxis, interpolate.splev(freqaxis, self.sidebandF, der = 0), lineStyle = '-', color = 'black')
        else:
            plt.plot(freqaxis, np.polynomial.polynomial.polyval(freqaxis, self.sidebandP), lineStyle = '-', color = 'black')
        #plt.ylim(6.*self.sidebandY.min() - 5.*self.sidebandY.max(), 6.*self.sidebandY.max() - 5.*self.sidebandY.min())
        plt.savefig(saveprefix + '_subtracted.png')
        plt.close()

        # pure signal
        plt.figure(2)
        plt.plot(self.signal.freq, self.signal.amp, '.-')
        plt.plot(self.signal.freq, np.zeros(self.signal.freq.size), '--', color = 'red')
        plt.plot(freqaxis_short, np.full(freqaxis_short.size, self.signal.peakY*0.5), '--', color = 'red')
        plt.savefig(saveprefix + '_signal.png')
        plt.close()

        # qcurve adjustment
        if not self.qcurveless:
            plt.figure(3)
            plt.plot(self.data.freq, self.data.amp, '.', color = 'red')
            plt.plot(self.ref.freq + self.xOffset, self.scale*(self.ref.amp + self.yOffset), '.', color = 'blue')
            plt.savefig(saveprefix + '_qadj.png')
            plt.close()

    def summary(self, filename):
        """Print the summary of the existing data at shutdown"""

        now = datetime.now()
        summaryPath = os.path.join(self.refPath.replace('config', '%02d' % now.day))
        summaryFile = os.path.join(summaryPath, filename)

        fout = open(summaryFile, 'w')
        for res in self.results:
            # format: sweepID, time1, time2, area, NMR temp, peakX, peakY, bkg-subtract quality
            fout.write('%d,%d,%d,%.4e,%.4f,%.4e,%.4e,%.4e,%.4e,%d\n' % (res.sweepID, res.timeH, res.timeL, res.integral, res.temp, res.peakX, res.peakY, res.HML, res.HMR, res.status))
        fout.close()

def recvall(conn, timeout):
    conn.setblocking(0)

    dataIn = ''
    beginT = time.time()
    while True:
        if '?' in dataIn:
            break
        elif time.time() - beginT > timeout:
            break

        try:
            data = conn.recv(8192)
            if len(data) > 0:
                dataIn = dataIn + data
                beginT = time.time()
            else:
                time.sleep(0.1)
        except:
            pass

    return dataIn.replace('?', '')

def main(options):
    # Initialize fast ana
    fastAna = NMRFastAna(options)

    # Initialize the telnet server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((options.host, options.port))
    except:
        logger.error('port %d is being used. Exit now.' % options.port)
        sys.exit(0)
    sock.listen(5)

    # Wait for the client to connect
    logger.info('server started, waiting for incoming data ...')
    while True:
        conn, client_address = sock.accept()
        logger.info('connected from', client_address[0])

        # wait for incoming data indefinitely
        idleCycle = 0
        while True:
            dataIn = recvall(conn, 1.)

            if 'shutdown' in dataIn.lower():
                logger.info('received shutdown command, closing ...')
                fastAna.summary('test.csv')
                sys.exit(0)

            if dataIn == '':
                idleCycle = idleCycle + 1
                if idleCycle > 10:  # disconnect if not received data for a long time
                    logger.warning('connection lost, waiting for another ...')
                    conn.shutdown(socket.SHUT_RDWR)
                    conn.close()
                    break
                continue

            if 'testtesttest' in dataIn.lower():
                logging.info('received test command, reply OK.')
                conn.sendall('ok?')
                continue

            idleCycle = 0  # reset idle cycle if received data again
            start_time = time.time()
            try:
                data = SweepData(dataIn)

                fastAna.setData(data)
                fastAna.setQcurve('auto')
            except:
                logger.critical('I/O ERROR')

                header, res = data.shortString()
                conn.sendall(header + res + '\n?')
                continue

            try:
                #fastAna.qCurveAdjust()
                fastAna.qCurveSubtract()
            except:
                logger.critical('Analysis ERROR')

                header, res = data.shortString()
                conn.sendall(header + res + '\n?')
                continue

            header, res = fastAna.runningAvg().shortString()
            conn.sendall(header + res + '\n?')

            finish_time = time.time()
            logger.info('finished one NMR analysis using %.2f ms' % ((finish_time - start_time)*1000))

if __name__ == '__main__':
    # parse the command line input
    parser = OptionParser('Usage: %prog [options]')
    parser.add_option('--host', type = 'string', dest = 'host', help = 'host ip address', default = 'localhost')
    parser.add_option('--port', type = 'int', dest = 'port', help = 'port number', default = 10000)
    parser.add_option('--log', type = 'string', dest = 'logfile', help = 'log file path', default = '')
    parser.add_option('--loglevel', type = 'string', dest = 'loglevel', help = 'log output level', default = 'info')
    parser.add_option('--mode', type = 'string', dest = 'mode', help = 'bkg subtraction method', default = 'spline')
    parser.add_option('--qcvless', action = 'store_true', dest = 'qcvless', help = 'Enable the qcurve less mode', default = False)
    parser.add_option('--qcvpath', type = 'string', dest = 'qcvpath', help = 'path where qcv file is stored', default = './')
    parser.add_option('--qcvfile', type = 'string', dest = 'qcvfile', help = 'default qcurve file', default = 'QCV3563809315.csv')
    (options, args) = parser.parse_args()

    logger.init(options)
    main(options)
