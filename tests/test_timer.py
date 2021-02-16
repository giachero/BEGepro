#!/usr/bin/env python

import glob
#from os.path import splitext
from threading import Timer

from os import listdir
from os.path import isfile, join
import signal, sys


def hello_world(mypath):
    Timer(5, hello_world, [mypath]).start() # called every minute
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    print(onlyfiles)

    
    return


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
    return 

def main():

    '''
    
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


    '''


    signal.signal(signal.SIGINT, signal_handler) 

    
    hello_world('./')

    
    return


if __name__ == '__main__':
    main()
