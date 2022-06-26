#!/usr/bin/env python

#./AvsEcompute.py --loc ~/work/tesi/data/ -m Std-232Th-3Bq-AEcalibration-im010421 -n 1 -d ~/work/tesi/data/ -r 1

#./AvsEcompute.py --loc ~/work/tesi/data/ -m 228Th-grafico-tesi-im260421_1 -n 265 -d ~/work/tesi/data/ -r 0

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
    exc_counter=0
    rt_obj=u.rise_time()
    peaks_obj=u.n_peaks()
    plateau_obj=u.plateau()
    nd_der_obj=u.second_derivative()
    simm_obj=u.simm()
    
    E=750 #energy in keV
    maxlim=7 #massimo nel segnale di corrente per il quale il segnale è ben definito

    print('\n+++++ START OF ANALYSIS +++++\n')
    
    cond=True

    for i in range(0,args.nfiles):  
        
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
           
            try:
                
                raw_wf            = np.array(data['trace'])      
                curr              = flt.curr_filter(raw_wf)               
                pulse_height      = data['pulseheight']
                energy            = data['energy']
                amplitude         = np.max(curr)
                avse              = amplitude / pulse_height
                risetime,t        = rt_obj.compute_rt2(raw_wf,4e-9)
                
                c,f               = nd_der_obj.compute_der(curr)
                             
                
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
                
                zeros_2der        = len(nd_der_obj.compute_n_zeros2(curr,t,c,f))                                                
                n_peaks           = len(peaks_obj.compute_n_peaks(curr,energy,E,maxlim))
                n_peaks_2der      = len(nd_der_obj.compute_n_peaks(f,t))
                
                
                simm              = simm_obj.compute_simm3(raw_wf)  
                                                             
                collector.add_pulse_height(pulse_height)
                collector.add_energy(energy)
                collector.add_amplitude(amplitude)
                collector.add_avse(avse)
                collector.add_risetime(risetime)
                collector.add_zeros_2der(zeros_2der)
                collector.add_n_peaks_2der(n_peaks_2der)
                collector.add_simm(simm)
                
                if(args.readTrace):
                    collector.add_curr(curr)
                    collector.add_trace(raw_wf)
                    
                collector.add_n_peaks(n_peaks)
                
                
            except:
                exc_counter+=1
                #print('exc: '+str(exc_counter)+' counter: '+str(counter))
                
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

        print('*** End of file ' + str(i+1) + '/' + str(args.nfiles) + ' ***')
                      
        if (cond):
            collector_tot=collector
            cond=False
        else:
            collector_tot+=collector
        
        collector_tot.remove_zeros()    
        collector_tot.update_index()
        
        if(((i%5==0) & (i!=0)) | (i==args.nfiles-1)):
            np.save(args.savedir + args.measname+'__'+str(i), collector_tot.get_parameters())
            
            if(args.readTrace):
                np.save(args.savedir + args.measname+"_trace"+'__'+str(i), collector_tot.get_traces())
                np.save(args.savedir + args.measname+"_curr"+'__'+str(i), collector_tot.get_curr())
            
            cond=True
            
    print('exc encountered: '+str(exc_counter))
    print('\n+++++ END OF ANALYSIS +++++\n')

    return


if __name__ == '__main__':
    main()
