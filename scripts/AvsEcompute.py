#!/usr/bin/env python

#./AvsEcompute.py --loc ~/work/data/ -m Std-232Th-3Bq-AEcalibration-im010421 -n 1 -d ~/work/data/

import numpy as np
import os
import sys
import math
import argparse


from begepro.rw import CAENhandler
from begepro.rw import CAENhandler_new
from begepro.dspro import filters as flt
from begepro.dspro import bege_event as be

def main():

    usage='./AvsEcompute.py -l /path/where/the/measurement/directory/is/ -m measurementName -n numberOfFiles'
    parser = argparse.ArgumentParser(description='Test script to read and analyze BEGe signals from CAEN desktop digitizer', usage=usage)

    parser.add_argument("-l", "--loc",    dest="dirloc",   type=str, help="Location of measurement directory", required = True)
    parser.add_argument("-m", "--meas",   dest="measname", type=str, help="Measurement name",                  required = True)
    parser.add_argument("-n", "--nfiles", dest="nfiles",   type=int, help="Number of files to analyze",        required = True)
    parser.add_argument("-d", "--dir",    dest="savedir",  type=str, help="Path where to save analysis",       required = True)
    
    args = parser.parse_args()

    path = args.dirloc + args.measname + '/FILTERED/DataF_CH1@DT5725SB_10806_' + args.measname

    counter = 0

    print('\n+++++ START OF ANALYSIS +++++\n')

    for i in range(args.nfiles):  
        
        print('*** Start of file ' + str(i+1) + '/' + str(args.nfiles) + ' ***')        
        
        filename=path+'.bin' if i==0 else str(i) + '.bin'
              
        ev_size, ev_numbers=CAENhandler_new.get_compass_size(filename,calibrated=True)
        
        collector=be.BEGeEvent(n_trace=ev_numbers,dim_trace=ev_size)
        
        rd = CAENhandler.compassReader(filename,calibrated=True)
               
        while True:

            data = rd.get()
            if data is None: break

            raw_wf       = np.array(data['trace'])
            curr         = flt.curr_filter(raw_wf)
            pulse_height = data['pulseheight']
            energy       = data['energy']
            amplitude    = np.max(curr)
            avse         = amplitude / pulse_height
                
            collector.add_pulse_height(pulse_height)
            collector.add_energy(energy)
            collector.add_amplitude(amplitude)
            collector.add_avse(avse)
            collector.add_trace(raw_wf)
            
            counter += 1
            if counter%10000 == 0: print('{sgn} signals processed...'.format(sgn=counter))

        print('*** End of file ' + str(i+1) + '/' + str(args.nfiles) + ' ***')
        collector.update_index()
        

    np.save(args.savedir + args.measname, collector.get_parameters())
    np.save(args.savedir + args.measname + "_trace", collector.get_traces())

    print('\n+++++ END OF ANALYSIS +++++\n')

    return


if __name__ == '__main__':
    main()
