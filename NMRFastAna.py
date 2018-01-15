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

def timestamp():
    """Create a timestamp for loggin purpose"""
    return datetime.now().strftime('%y%m%d-%H%M')

def parseSweepFile(filename, full = False):
    """parse an entire input file. if it's not an AVE file produced by LabView, read the entire file and make average"""
    assert os.path.exists(filename) or os.path.exists(filename.replace('AVE', '')), 'File %s does not exist.' % filename
    if 'AVE' in filename and os.path.exists(filename):
        return SweepData('\n'.join([line.strip() for line in open(filename)]))
    else: # if AVE file does not exist, try the original one and calculate the average myself ...
        filename = filename.replace('AVE', '')

        contents = [line.strip() for line in open(filename)]

        index = 0
        rsweeps = []
        while index < len(contents):
            nSteps = int(float(contents[index + 14]))
            rsweeps.append(contents[index:index + 35 + nSteps])
            index = index + 35 + nSteps

        if full:
            return [SweepData('\n'.join(rs)) for rs in rsweeps]

        sweep = SweepData('\n'.join(rsweeps[0]))
        for rs in rsweeps[1:]:
            sweep.amp = sweep.amp + SweepData('\n'.join(rs)).amp
        sweep.amp = sweep.amp/len(rsweeps)

        return sweep

class SweepData:
    """Container of sweep data"""
    #gain_value = [1., 20., 200.]
    gain_value = [1., 1., 1.]

    def __init__(self, inputStr = '', qfile = '', ts = 0, center = 213., freqMin = 212.6, freqMax = 213.4, nSteps = 500, gain = 1, amps = []):
        """parse the input string and fill the header+data info"""
        if inputStr == '':  # if nothing is provided, put together a fake header
            fakeHeader = ['' for i in range(35)]
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
        try:
            self.signalL    = float(info[8])
            self.signalH    = float(info[9])
        except:
            self.signalL    = 212.75
            self.signalH    = 213.25

        # indirect ones
        self.minFreq = self.centerFreq - (self.nSteps - 1.)/2.*self.stepSize
        self.maxFreq = self.centerFreq + (self.nSteps - 1.)/2.*self.stepSize

        # x and y data
        self.freq = np.linspace(self.minFreq, self.maxFreq, num = self.nSteps, endpoint = True)
        self.amp  = np.array([float(line)/SweepData.gain_value[self.gain] for line in info[35:]])

        self.peakIdx = np.argmax(self.amp)
        self.peakX = self.freq[self.peakIdx]
        self.peakY = self.amp[self.peakIdx]*SweepData.gain_value[self.gain]
        
        # basic sanity check
        assert self.amp.size == self.nSteps, 'step size in header does not match data'
        assert self.nSteps > 50, 'number of steps too small'
        assert self.gain <= 2, 'gain selection wrong'

        # spline representation
        self.func = None

        # integral of the curve
        self.integral = 0.

        # flag of bkg-subtraction status, 0 for fail, 1 for success
        self.status = 0

    def update(self, avgwin = 3):
        """update the peak info using the most recent data"""
        self.peakIdx = np.argmax(self.amp)
        peakSlice = np.arange(self.peakIdx-avgwin, self.peakIdx+avgwin+1, 1)

        self.peakX = (self.freq[peakSlice]*self.amp[peakSlice]).sum()/self.amp[peakSlice].sum()
        self.peakY = np.interp(self.peakX, self.freq, self.amp)*SweepData.gain_value[self.gain]

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

    def __init__(self, options):
        self.freqCenter = 213.
        self.freqWin = 0.25
        self.freqMin = 212.65
        self.freqMax = 213.35
        self.freqAdjMin = self.freqMin
        self.freqAdjMax = self.freqMax
        self.sampleSize = 0.002

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
        self.refFile = ''
        self.refPath = options.qcvpath
        self.refDefaultFile = options.qcvfile

        # q curve subtracted data
        self.subtractedX = None
        self.subtractedY = None
        self.subtractedF = None

        # subset of q qurve subtracted data, only in the pre-defined sideband region
        self.sidebandX = None
        self.sideBandY = None
        self.sidebandF = None   # spline function for sideband
        self.sidebandP = None   # polymonial function for sideband

        # bkg-subtraction mode
        self.mode = options.mode

        # pure signal after furture background subtraction with spline, in the form of SweepData
        self.signal = None

        # container of the results so far
        self.results = []

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

        # check if either AVE file or original file exists
        qfileAbs = os.path.join(self.refPath, qfile)
        assert (os.path.exists(qfileAbs) or os.path.exists(qfileAbs.replace('AVE', ''))), 'Q curve input file %s does not exist.' % qfileAbs

        if self.refFile != qfile:
            self.refFile = qfile
            self.ref = parseSweepFile(qfileAbs)
            self.ref.interpolate()

    def setData(self, dataIn):
        """set the data input"""
        self.data = dataIn
        self.data.interpolate()

        self.xOffset = 0.
        self.yOffset = 0.

        self.freqCenter = 0.5*(self.data.signalL + self.data.signalH)
        self.freqwin    = 0.5*(self.data.signalH - self.data.signalL)

        # check if it's from a new run
        if len(self.results) > 0 and self.data.sweepID < self.results[-1].sweepID:
            self.results = []

    def range(self):
        """adjust the range according to the current offset parameter"""
        # find the real minimum and maximum given the offset
        tempmin = self.data.freq[0] + self.xOffset
        tempmax = self.data.freq[-1]
        if self.xOffset < 0:
            tempmin = self.data.freq[0]
            tempmax = self.data.freq[-1] + self.xOffset
        return (max(tempmin, self.freqMin), min(tempmax, self.freqMax))

    def chisq(self, x, y):
        """definition of the chi^2 function for Q curve adjustment"""
        self.xOffset = x
        self.yOffset = y
        self.freqAdjMin, self.freqAdjMax = self.range()
        nSamples = int((self.freqAdjMax - self.freqAdjMin)/self.sampleSize)

        freq = np.array([f for f in np.linspace(self.freqAdjMin, self.freqAdjMax, num = nSamples, endpoint = False) if abs(f - self.freqCenter) > self.freqWin])
        chi2 = ((interpolate.splev(freq, self.data.func, der = 0) - (interpolate.splev(freq - x, self.ref.func, der = 0) + y))**2).sum()/freq.size
        
        return chi2

    def qCurveAdjust(self):
        """use sideband region for the Q curve adjustment"""
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

    def gaus(self, x, A, m, s):
        return A*np.exp(-(x - m)**2/(2.*s**2))

    def pol1(self, x, p0, p1):
        return p0 + p1*x

    def pol2(self, x, p0, p1, p2):
        return p0 + p1*x + p2*x*x

    def pol3(self, x, p0, p1, p2, p3):
        return p0 + p1*x + p2*x*x + p3*x*x*x

    def gausfit(self, xdata, ydata):
        pfit, perr = curve_fit(self.gaus, xdata, ydata, p0 = [ydata.max(), self.freqCenter, self.freqWin])
        return pfit[1], pfit[2]

    def polfit(self, xdata, ydata):
        chi2 = [999. for i in range(3)]
        pfit1, _ = curve_fit(self.pol1, xdata, ydata, p0 = [0., 0.])
        chi2[0] = ((ydata - np.polynomial.polynomial.polyval(xdata, pfit1))**2).sum()#/(xdata.size - 2)

        pfit2, _ = curve_fit(self.pol2, xdata, ydata, p0 = [0., 0., 0.])
        chi2[1] = ((ydata - np.polynomial.polynomial.polyval(xdata, pfit2))**2).sum()#/(xdata.size - 3)
        
        pfit3, _ = curve_fit(self.pol3, xdata, ydata, p0 = [0., 0., 0., 0.])
        chi2[2] = ((ydata - np.polynomial.polynomial.polyval(xdata, pfit3))**2).sum()#/(xdata.size - 4)

        index = chi2.index(min(chi2))
        return [pfit1, pfit2, pfit3][index], min(chi2)

    def qCurveSubtract(self):
        """with the adjusted Q curve info, make the Q curve subtraction and then the sideband spline subtraction"""
        self.freqAdjMin, self.freqAdjMax = self.range()

        subtractedSlice = np.array([i for i in range(self.data.freq.size) if self.data.freq[i] > self.freqAdjMin and self.data.freq[i] < self.freqAdjMax])
        self.subtractedX = self.data.freq[subtractedSlice]
        self.subtractedY = self.data.amp[subtractedSlice] - interpolate.splev(self.subtractedX - self.xOffset, self.ref.func, der = 0) - self.yOffset
        #self.subtractedF = interpolate.splrep(self.subtractedX, self.subtractedY, s = 0)

        sampleRate = int(self.sampleSize/self.data.stepSize)
        if sampleRate < 1:
            sampleRate = 1
        adjustedCenter = self.freqCenter  #self.subtractedX[np.argmax(self.subtractedY)]
        RsidebandSliceL  = np.array([i for i in range(0, self.subtractedX.size, sampleRate) if adjustedCenter - self.subtractedX[i] > self.freqWin])
        RsidebandSliceR  = np.array([i for i in range(0, self.subtractedX.size, sampleRate) if self.subtractedX[i] - adjustedCenter > self.freqWin])

        sidebandMeanL = self.subtractedY[RsidebandSliceL].mean()
        sidebandSigmaL = self.subtractedY[RsidebandSliceL].std()
        sidebandMeanR = self.subtractedY[RsidebandSliceR].mean()
        sidebandSigmaR = self.subtractedY[RsidebandSliceR].std()

        sidebandSliceL = np.array([i for i in range(0, self.subtractedX.size, sampleRate) if adjustedCenter - self.subtractedX[i] > self.freqWin and abs(self.subtractedY[i] - sidebandMeanL) < 2.*sidebandSigmaL])
        sidebandSliceR = np.array([i for i in range(0, self.subtractedX.size, sampleRate) if self.subtractedX[i] - adjustedCenter > self.freqWin and abs(self.subtractedY[i] - sidebandMeanR) < 2.*sidebandSigmaR])
        sidebandSlice = np.concatenate((sidebandSliceL, sidebandSliceR))

        self.sidebandX = self.subtractedX[sidebandSlice]
        self.sidebandY = self.subtractedY[sidebandSlice]

        self.signal = copy.deepcopy(self.data)
        if self.mode == 'spline':
            self.sidebandF = interpolate.splrep(self.sidebandX, self.sidebandY, s = 1, k = 3)
            centerY = self.subtractedY - interpolate.splev(self.subtractedX, self.sidebandF, der = 0)
        else:
            self.sidebandP, _ = self.polfit(self.sidebandX, self.sidebandY)
            centerY = self.subtractedY - np.polynomial.polynomial.polyval(self.subtractedX, self.sidebandP)

        paddingL = np.array([f for f in self.data.freq if f < self.subtractedX.min()])
        paddingH = np.array([f for f in self.data.freq if f > self.subtractedX.max()])
        self.signal.freq = np.concatenate((paddingL, self.subtractedX, paddingH))
        self.signal.amp = np.concatenate((np.full(paddingL.size, 0.), centerY, np.full(paddingH.size, 0.)))
        self.signal.update()

        signalSlice = np.array([i for i in range(self.subtractedX.size) if abs(self.subtractedX[i] - adjustedCenter) < self.freqWin])
        self.signal.integral = self.signal.amp[signalSlice].sum()*self.signal.stepSize

        # check the status of the bkg-subtraction
        sidebandSlopeL = (self.subtractedY[sidebandSliceL[-1]] - self.subtractedY[sidebandSliceL[0]])/(self.subtractedX[sidebandSliceL[-1]] - self.subtractedX[sidebandSliceL[0]])
        sidebandSlopeR = (self.subtractedY[sidebandSliceR[-1]] - self.subtractedY[sidebandSliceR[0]])/(self.subtractedX[sidebandSliceR[-1]] - self.subtractedX[sidebandSliceR[0]])
        if abs(sidebandSlopeL - sidebandSlopeR)/(abs(sidebandSlopeL) + abs(sidebandSlopeR)) < 1.:
            self.signal.status = 2
        else:
            self.signal.status = 1

        self.results.append(self.signal)

    def plot(self, path, prefix):
        saveprefix = os.path.join(path, prefix)
        freqaxis = np.linspace(self.freqAdjMin, self.freqAdjMax, num = 1000, endpoint = True)

        # raw data + raw q curve
        plt.figure(0)
        plt.plot(self.data.freq, self.data.amp, 'o', color = 'red')
        plt.plot(self.ref.freq, self.ref.amp, 'o', color = 'blue')
        plt.savefig(saveprefix + '_raw.png')
        plt.close()

        # bkg subtraction overlay -- focus on bkg
        plt.figure(1)
        plt.plot(self.subtractedX, self.subtractedY, 'o')
        plt.plot(self.sidebandX, self.sidebandY, 'o', color = 'red')
        if self.mode == 'spline':
            plt.plot(freqaxis, interpolate.splev(freqaxis, self.sidebandF, der = 0), lineStyle = '-', color = 'black')
        else:
            plt.plot(freqaxis, np.polynomial.polynomial.polyval(freqaxis, self.sidebandP), lineStyle = '-', color = 'black')
        #plt.ylim(6.*self.sidebandY.min() - 5.*self.sidebandY.max(), 6.*self.sidebandY.max() - 5.*self.sidebandY.min())
        plt.savefig(saveprefix + '_subtracted.png')
        plt.close()

        # pure signal
        plt.figure(2)
        plt.plot(self.signal.freq, self.signal.amp, 'o-')
        plt.plot(self.signal.freq, np.zeros(self.signal.freq.size), '--', color = 'red')
        plt.savefig(saveprefix + '_signal.png')
        plt.close()

        # qcurve adjustment
        plt.figure(3)
        plt.plot(self.data.freq, self.data.amp, 'o', color = 'red')
        plt.plot(self.ref.freq + self.xOffset, self.scale*(self.ref.amp + self.yOffset), 'o', color = 'blue')
        plt.savefig(saveprefix + '_qadj.png')
        plt.close()

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

    #print len(dataIn), time.time() - beginT
    return dataIn.replace('?', '')

def main(options):
    # Initialize fast ana
    fastAna = NMRFastAna(options)

    # Initialize the telnet server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((options.host, options.port))
    except:
        print timestamp(), 'WARNING: port %d is being used. Exit now.' % options.port
        sys.exit(0)
    sock.listen(5)

    # Wait for the client to connect
    print timestamp(), 'INFO: server started, waiting for incoming data ...'
    while True:
        conn, client_address = sock.accept()
        print timestamp(), 'INFO: connected from', client_address[0]

        # wait for incoming data indefinitely
        while True:
            dataIn = recvall(conn, 1.)

            if 'shutdown' in dataIn.lower():
                print timestamp(), 'INFO: received shutdown command, closing ...'
                sys.exit(0)

            if dataIn == '':
                print timestamp(), 'WARNING: connection lost, waiting for another ...'
                break

            start_time = time.time()
            try:
                data = SweepData(dataIn)

                fastAna.setData(data)
                fastAna.setQcurve('auto')
            except Exception, err:
                print timestamp(), 'I/O Error: ', err
                
                header, res = data.shortString()
                conn.sendall(header + res + '\n?')
                continue

            fastAna.freqWin = 0.25
            if 'pol' in dataIn.lower():
                fastAna.freqWin = 0.2

            try:
                #fastAna.qCurveAdjust()
                fastAna.qCurveSubtract()
            except Exception, err:
                print timestamp(), 'Analysis Error: ', err
                
                header, res = data.shortString()
                conn.sendall(header + res + '\n?')
                continue

            header, res = fastAna.runningAvg().shortString()
            conn.sendall(header + res + '\n?')

            finish_time = time.time()
            print timestamp(), 'INFO: finished one NMR analysis using %.2f ms' % ((finish_time - start_time)*1000)

if __name__ == '__main__':
    # parse the command line input
    parser = OptionParser('Usage: %prog [options]')
    parser.add_option('--host', type = 'string', dest = 'host', help = 'host ip address', default = 'localhost')
    parser.add_option('--port', type = 'int', dest = 'port', help = 'port number', default = 10000)
    parser.add_option('--log', type = 'string', dest = 'log', help = 'log file', default = '')
    parser.add_option('--mode', type = 'string', dest = 'mode', help = 'bkg subtraction method', default = 'spline')
    parser.add_option('--qcvpath', type = 'string', dest = 'qcvpath', help = 'path where qcv file is stored', default = './')
    parser.add_option('--qcvfile', type = 'string', dest = 'qcvfile', help = 'default qcurve file', default = 'QCV3563809315.csv')
    (options, args) = parser.parse_args()

    main(options)
