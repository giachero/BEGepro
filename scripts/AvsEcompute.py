#!/usr/bin/env python

#./AvsEcompute.py --loc ~/work/tesi/data/ -m Std-232Th-3Bq-AEcalibration-im010421 -n 1 -d ~/work/tesi/data/ -r 1

import numpy as np
import os
import sys
import math
import argparse

import IPython 



from begepro.rw import CAENhandler
from begepro.rw import CAENhandler_new
from begepro.dspro import filters as flt
from begepro.dspro import bege_event as be
from begepro.dspro import utils as u

def main():

    usage='./AvsEcompute.py -l /path/where/the/measurement/directory/is/ -m measurementName -n numberOfFiles -r readTrace'
    parser = argparse.ArgumentParser(description='Test script to read and analyze BEGe signals from CAEN desktop digitizer', usage=usage)

    parser.add_argument("-l", "--loc",    dest="dirloc",   type=str, help="Location of measurement directory", required = True)
    parser.add_argument("-m", "--meas",   dest="measname", type=str, help="Measurement name",                  required = True)
    parser.add_argument("-n", "--nfiles", dest="nfiles",   type=int, help="Number of files to analyze",        required = True)
    parser.add_argument("-d", "--dir",    dest="savedir",  type=str, help="Path where to save analysis",       required = True)
    parser.add_argument("-r", "--read",   dest="readTrace",type=int, help="Read also traces and currents",     required = True)
    
    args = parser.parse_args()
    
    print(args)

    path = args.dirloc + args.measname + '/FILTERED/DataF_CH1@DT5725SB_10806_' + args.measname

    counter = 0
    rt_obj=u.rise_time()
    peaks_obj=u.n_peaks()
    plateau_obj=u.plateau()
    nd_der_obj=u.second_derivative()
    
    E=750 #energy in keV
    maxlim=7 #massimo nel segnale di corrente per il quale il segnale è ben definito

    print('\n+++++ START OF ANALYSIS +++++\n')

    for i in range(args.nfiles):  
        
        print('*** Start of file ' + str(i+1) + '/' + str(args.nfiles) + ' ***')        
        
        filename=path+'.bin' if i==0 else path+'_'+str(i) + '.bin'
              
        ev_size, ev_numbers=CAENhandler_new.get_compass_size(filename,calibrated=True)
        
        collector=be.BEGeEvent(n_trace=ev_numbers,
                               dim_trace=ev_size,
                               trace=np.array([]) if args.readTrace==0 else None,
                               curr=np.array([])  if args.readTrace==0 else None)

        rd = CAENhandler.compassReader(filename,calibrated=True)
               
        while True:

            data = rd.get()
            if data is None: break

            raw_wf            = np.array(data['trace'])      
            curr              = flt.curr_filter(raw_wf)               
            pulse_height      = data['pulseheight']
            energy            = data['energy']
            amplitude         = np.max(curr)
            avse              = amplitude / pulse_height
            risetime,t        = rt_obj.compute_rt(raw_wf,4e-9)
            zeros_2der        = nd_der_obj.compute_n_zeros(raw_wf,t)               
                
            n_peaks      = len(peaks_obj.compute_n_peaks(curr,energy,E,maxlim))
                
            collector.add_pulse_height(pulse_height)
            collector.add_energy(energy)
            collector.add_amplitude(amplitude)
            collector.add_avse(avse)
            collector.add_risetime(risetime)
            collector.add_zeros_2der(len(zeros_2der))
            
            if(args.readTrace):
                collector.add_curr(curr)
                collector.add_trace(raw_wf)
                
            collector.add_n_peaks(n_peaks)
            
            #if counter==8: IPython.embed()
            
            
            counter += 1
            
            if counter%10000 == 0: print('{sgn} signals processed...'.format(sgn=counter))

        print('*** End of file ' + str(i+1) + '/' + str(args.nfiles) + ' ***')
                      
        if i==0:
            collector_tot=collector
        else:
            collector_tot+=collector
            
        collector_tot.update_index()
        
    np.save(args.savedir + args.measname, collector_tot.get_parameters())
    np.save(args.savedir + args.measname+"_trace", collector_tot.get_traces())
    np.save(args.savedir + args.measname+"_curr", collector_tot.get_curr())

    print('\n+++++ END OF ANALYSIS +++++\n')

    return


if __name__ == '__main__':
    main()
