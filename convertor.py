import os
import sys
from optparse import OptionParser

import NMRFastAna

parser = OptionParser('Usage: %prog [options]')
parser.add_option('-i', type = 'string', dest = 'input', help = 'input file', default = '')
parser.add_option('-o', type = 'string', dest = 'output', help = 'output file', default = '')
parser.add_option('-q', type = 'string', dest = 'qcv', help = 'qcurve file specified in header', default = '')
(options, args) = parser.parse_args()

# read input files
inputs = [line.strip() for line in open(options.input)]

# save
outputf = open(options.output, 'w')
for i in inputs:
    contents = [a.strip() for a in i.split(',')]
    s = NMRFastAna.SweepData(qfile = options.qcv, ts = int(contents[0]), amps = contents[1:])

    outputf.write(s.longString() + '\n')
outputf.close()
