#!/usr/bin/env python

#./AvsEcompute.py --loc ~/work/tesi/data/ -m Std-232Th-3Bq-AEcalibration-im010421 -n 1 -d ~/work/tesi/data/ -r 1

#./AvsEcompute.py --loc ~/work/tesi/data/ -m 228Th-grafico-tesi-im260421_1 -n 265 -d ~/work/tesi/data/ -r 0
#python3 AvsEcompute.py --loc ~/work/Data/Raw/ -m CRC01-Zebru-im22012021 -n 265 -d ~/work/Data/Analized -rT 1 -rC 1 -c 0
#python3 AvsEcompute.py --loc ~/work/Data/Raw/ -m 228Th-grafico-tesi-im260421_1 -n 265 -d ~/work/Data/Analized -rT 1 -rC 1 -c 1

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
from begepro.dspro import utils2 as u
import matplotlib.pyplot as plt

def main():

    usage='./AvsEcompute.py -l /path/where/the/measurement/directory/is/ -m measurementName -n numberOfFiles -rT readTrace -rC readCurr -c calibrated'
    parser = argparse.ArgumentParser(description='Test script to read and analyze BEGe signals from CAEN desktop digitizer', usage=usage)

    parser.add_argument("-l", "--loc",    dest="dirloc",     type=str, help="Location of measurement directory", required = True)
    parser.add_argument("-m", "--meas",   dest="measname",   type=str, help="Measurement name",                  required = True)
    parser.add_argument("-n", "--nfiles", dest="nfiles",     type=int, help="Number of files to analyze",        required = True)
    parser.add_argument("-d", "--dir",    dest="savedir",    type=str, help="Path where to save analysis",       required = True)
    parser.add_argument("-rT", "--readT",   dest="readTrace",type=int, help="Read also traces ",                 required = True)
    parser.add_argument("-rC", "--readC",   dest="readCurr", type=int, help="Read also currents",                required = True)
    parser.add_argument("-c", "--calib",  dest="calibrated", type=int, help="Calibrated in energy",              required = True)
    
    args = parser.parse_args()
    
    print(args)

    path = args.dirloc + args.measname + '/FILTERED/DataF_CH1@DT5725SB_10806_' + args.measname

    counter = 0
    exc_counter=0
    rt_obj=u.rise_time()
    nd_der_obj=u.second_derivative()
    simm_obj=u.simm()

    print('\n+++++ START OF ANALYSIS +++++\n')

    for i in range(0,args.nfiles): 
        
        print('*** Start of file ' + str(i+1) + '/' + str(args.nfiles) + ' ***')        
        
        filename=path+'.bin' if i==0 else path+'_'+str(i) + '.bin'
        ev_size, ev_numbers=CAENhandler_new.get_compass_size(filename,args.calibrated)

        collector=be.BEGeEvent(n_trace=ev_numbers,
                               dim_trace=ev_size,
                               trace=np.array([]) if args.readTrace==0 else None,
                               curr=np.array([])  if args.readTrace==0 else None)

        rd = CAENhandler.compassReader(filename,args.calibrated)
         
        while True: 
            data = rd.get()
            if data is None: break
                  
            try:
                
                raw_wf            = np.array(data['trace'])   
                curr              = flt.curr_filter(raw_wf)          
                pulse_height      = data['pulseheight']
                energy            = data['energy'] if args.calibrated else 0
                amplitude         = np.max(curr)
                avse              = amplitude / pulse_height 
                
                if(args.readTrace): collector.add_trace(raw_wf)
                if(args.readCurr): collector.add_curr(curr)

                plt.figure()
                plt.plot(collector.get_traces()[i])
                plt.show()
                
                curr_norm=normalize(curr)
                raw_wf_norm=normalize(raw_wf)
                risetime,t        = rt_obj.compute_rt(raw_wf_norm,4e-9)
                f, smooth         = nd_der_obj.compute_der(curr)
                rt2,t2            = rt_obj.compute_rt(raw_wf_norm,4e-9,riseTimeLimits=(0.05,0.95))            
                
                #print(counter)
                """
                if counter==57:
                    import pylab as plt
                    plt.figure()
                    plt.plot(raw_wf)
                    print(t[0])
                    plt.scatter(rt_obj.prova,raw_wf[rt_obj.prova])
                    plt.show()
                """
            
                zeros_2der        = len(nd_der_obj.compute_n_zeros(f, t))                                                
                n_peaks           = -1 #len(peaks_obj.compute_n_peaks(curr,energy,E,maxlim))
                           
                n_peaks_2der      = len(nd_der_obj.compute_n_peaks(f,t2))
                simm              = simm_obj.compute_simm2(raw_wf_norm,f)         
                area              = nd_der_obj.compute_area(normalize(f))                         
                collector.add_pulse_height(pulse_height)
                
                collector.add_energy(energy)
                collector.add_amplitude(amplitude)
                collector.add_avse(avse)
                collector.add_risetime(risetime)
                collector.add_zeros_2der(zeros_2der)
                collector.add_n_peaks_2der(n_peaks_2der)
                collector.add_simm(simm)
                collector.add_area(area)
                   
                collector.add_n_peaks(n_peaks) #len(peaks_obj.compute_n_peaks2(curr,rt_obj.compute_rt2(raw_wf,4e-9,(0.005,0.995))[1])))
                    
            except:
                exc_counter+=1
                print('exc: ', exc_counter)
                
                """
                import pylab as plt
                plt.figure()
                plt.plot(raw_wf)
                print(t)
                plt.scatter(rt_obj.prova,raw_wf[rt_obj.prova])            
                plt.savefig('/home/marco/work/tesi/images/exceptions/img_'+str(counter)+'.png')"""
            
            #if counter==22: break #IPython.embed()
                  
            counter += 1
            
            if counter%10000 == 0: print('{sgn} signals processed...'.format(sgn=counter))
            #if counter==10000: break

        print('*** End of file ' + str(i+1) + '/' + str(args.nfiles) + ' ***')
        
        collector.remove_zeros()    
        collector.update_index()
    
        np.save(args.savedir + args.measname+'__'+str(i), collector.get_parameters())
        
        if(args.readTrace): np.save(args.savedir + args.measname+"_trace"+'__'+str(i), collector.get_traces())
        if(args.readCurr): np.save(args.savedir + args.measname+"_curr"+'__'+str(i), collector.get_curr())
        print('*** Saved file ' + str(i+1) + '/' + str(args.nfiles) + ' ***')
            
    print('exc encountered: '+str(exc_counter))
    print('\n+++++ END OF ANALYSIS +++++\n')

    return

def normalize(x):
    x=np.array(x).astype(np.float64)
    x=x-min(x)
    x=x/max(x)
    return x
    
if __name__ == '__main__':
    main()