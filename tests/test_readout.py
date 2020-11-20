#!/usr/bin/env python

import numpy as np
import pylab as plt
import IPython,os
import argparse

from rw import DT5751read


def main():

    usage='./test_readout.py --file /path/where/the/file/is/filenane'
    parser = argparse.ArgumentParser(description='Test script to read data from xml files created by the CAEN DT5751 digitizer', usage=usage)

    parser.add_argument("-f", "--file", dest="ifile", help="Input file",          required = True)
    parser.add_argument("-n", "--numb", dest="numb",  help="Number of read file", required = True)
    args = parser.parse_args()

    
    dr=DT5751read.DT5751reader(filename) 

    Mm=list()

    ch=1
    
    while True:
        #for i in range(0, 10000):
        data=dr.get()
        if data is None: #or len(Mm)==100:
            break
        
        if ch in data['channels']:
            Mm.append(np.max(data['channels'][ch])-np.min(data['channels'][ch]))


    print (len(Mm))

    np.savetxt(os.path.splitext(filename)[0]+'_Mn.dat', Mm, fmt='%10.3f')
    
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
