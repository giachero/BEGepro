#!/usr/bin/env python

import numpy as np
import pylab as plt
import IPython, os, sys
import argparse

from begepro.rw import CAENhandler,  H5Handler


def main():

    #Check for arguments
    if len(sys.argv[1:]) == 0:
        parser.error('No argument given!')
        return 

    usage='./test_conversion.py --file /path/where/the/file/is/filenane'
    parser = argparse.ArgumentParser(description='Test script to read data from xml files created by the CAEN desktop digitizer', usage=usage)

    parser.add_argument("-f", "--file"   , dest="ifile", type=str,  help="Input file",                required = True)
    #parser.add_argument("-c", "--ch"     , dest="ch",    type=int,  help="Channel to analyze",        default  = 0)
    #parser.add_argument("-n", "--nevents", dest="nevs",  type=int,  help="First n events to analyze", default  = None)
    
    args = parser.parse_args()

    dr=CAENhandler.compassReader(args.ifile)
    labels = list(dr.get().keys())    

    #['board', 'channel', 'ttag', 'energy', 'flag', 'evsize', 'trace']

    
    dw = H5Handler.H5Writer(labels, os.path.splitext(args.ifile)[0]+'.hdf5')
    bsize=1000
    dw.set_buffer_size(bsize)
    dw.set_compression(9)
    
    dw.set_data_type(dict(zip(['board', 'channel', 'ttag', 'energy', 'flag', 'evsize', 'trace'],
                              [np.int16, np.int16, np.int64, np.int16, np.uint16, np.uint16, np.int16])))


    cnt=0
    while True:
        data=dr.get()
        if data is None: # or (N is not None and len(Mm)==N): 
            break

        for l in labels: #['energy', 'trace']:
            if l not in ['energy', 'trace']:
                del data[l]
        
        dw.add_data(data)
        cnt+=1
        
        if not cnt % (bsize):
            print(" Number of read events: %03d"%cnt)
        
    IPython.embed(banner1="")
    return


if __name__ == '__main__':
    main()
