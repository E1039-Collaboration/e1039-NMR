from NMRFastAna import SweepData, NMRFastAna

analyzer = None

def process(dataIn, qcvPath):

    global analyzer
    
    # Initialize the analyzer if it's not inited yet
    if analyzer is None:
        analyzer = NMRFastAna()
        analyzer.refPath = qcvPath

    if len(dataIn) < 5:
        return '?'
    elif 'testtesttest' in dataIn.lower():
        return 'ok?'
    elif 'shutdown' in dataIn.lower():
        analyzer = None
        return 'shutdown?'
    else:
        dataIn = dataIn.replace('?', '')
        try:
            data = SweepData(dataIn)
            data.status = 0
        except:
            return dataIn+'?'
        
        try:
            analyzer.setData(data)
            analyzer.setQcurve('auto')
        except:
            header, res = data.shortString()
            return header+res+'\n?'
        
        try:
            analyzer.qCurveSubtract()
        except:
            header, res = data.shortString()
            return header+res+'\n?'
        
        header, res = analyzer.runningAvg().shortString()
        return header+res+'\n?'

    