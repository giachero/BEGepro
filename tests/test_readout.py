#!/usr/bin/env python

import numpy as np
import pylab as plt
import IPython, os, sys
import argparse

from begepro.rw import CAENhandler


def main():

    # Check for arguments
    if len(sys.argv[1:]) == 0:
        parser.error('No argument given!')
        return 

    usage='./test_readout.py --file /path/where/the/file/is/filenane'
    parser = argparse.ArgumentParser(description='Test script to read data from xml files created by the CAEN desktop digitizer', usage=usage)

    parser.add_argument("-f", "--file"   , dest="ifile", type=str,  help="Input file",                required = True)
    parser.add_argument("-c", "--ch"     , dest="ch",    type=int,  help="Channel to analyze",        default  = 0)
    parser.add_argument("-n", "--nevents", dest="nevs",  type=int,  help="First n events to analyze", default  = None)
    
    args = parser.parse_args()

    
    dr=CAENhandler.XMLreader(args.ifile) 

    Mm=list()

    ch=1
    
    while True:
        data=dr.get()
        if data is None or (args.nevs is not None and len(Mm)==args.nevs): #or len(Mm)==100:
            break
        
        if ch in data['channels']:
            Mm.append(np.max(data['channels'][args.ch])-np.min(data['channels'][args.ch]))


    print (len(Mm))

    np.savetxt(os.path.splitext(args.ifile)[0]+'_Mn.dat', Mm, fmt='%10.3f')
    
    #plt.plot(data[ch])
                

    '''
    
    nbins=2**10
    
    plt.clf()
    plt.hist(Mm, bins=nbins, density=False, alpha=0.7, rwidth=1, label='Max-Min')
    plt.title('Amplitudes Distribution: $n_{bins} = %d$'%nbins )
    plt.xlabel('Amplitude')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()


    '''
    
    IPython.embed(banner1="")
    return


if __name__ == '__main__':
    main()
