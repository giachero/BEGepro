#!/usr/bin/env python

from begepro.rw import CAENhandler_new  as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split 
from sklearn import mixture
import matplotlib.colors as clr
from begepro.dspro import bege_event as be


import psutil
import random

import math

import pickle

import IPython 
from begepro.dspro import histfit as hf
from begepro.dspro import utils_analysis as ua

a=-0.090383
b=0.20574
calVec = [a + b*i for i in range(2**14+1)]
crioconite=False

def main():

    #OPEN FILES

    print('\n')
    print('\n')
    if(crioconite):
        Emax=700
        Emin=300
    else:
        Emax=2700
        Emin=150
    
    
    for i in range(5,45,5): #45 for crioconite
        if(crioconite):
            filename='/home/marco/work/tesi/data/NewParameters/Crioconite/NoTrace/CRC01-Zebru-im22012021__'+str(i)+'.npy'
        else:
            filename='/home/marco/work/tesi/data/NewParameters/228Th/NoTrace/228Th-grafico-tesi-im260421_1__'+str(i)+'.npy'
        coll=ca.NPYreader(filename,False).get_event()
        if(crioconite):
            coll.calibrate((a,b))
        if(i==5):
            coll_tot=coll.subset('energy',Emin,Emax)
        else:
            coll_tot=coll_tot+coll.subset('energy',Emin,Emax)
            del(coll)
            
        print('opened '+str(i)+' , Ram: '+str(psutil.virtual_memory()[2]))
        
    coll_tot=coll_tot.subset('energy',Emin,Emax).subset('energy',index=np.arange(0,1500000))
    
    if(crioconite):
        #Crioconite
        comparison_energy=(605,615)
        peak={  'full-energy' :(660,670)}
        compton=(355,395)
        
        ae_cut=0.01441559398334617
    else:
        #228Th
        comparison_energy=(2100,2110)
        #comparison_energy=(580,590)
        peak={  'double_escape' :(1590,1600),
                'peakBi'        :(1620,1625),         
                'full-energy'        :(2605,2620)}
        compton=(1850,2000)
        
        ae_cut=0.01805600127213223
    
    #SUP
    name = '/home/marco/work/tesi/data/selected_adc_channels_Cr.npy' if crioconite else '/home/marco/work/tesi/data/selected_adc_channels_Th.npy'
    comp = np.load(name)
    
    collSup=be.BEGeEvent(coll_tot.n_trace,coll_tot.dim_trace,pheight=comp)
    collSup.calibrate((a,b))
    
    
    #UNSUP 
    name = '/home/marco/work/tesi/data/adc_Cr.npy' if crioconite else '/home/marco/work/tesi/data/adc_Th.npy'
    comp = np.load(name)
    
    collUnsup=be.BEGeEvent(coll_tot.n_trace,coll_tot.dim_trace,pheight=comp)
    collUnsup.calibrate((a,b))
    
    
    #peak compton
    plt.figure()
    
    calVecR=np.array(calVec)
    calVecR=calVecR[range(0,calVecR.shape[0],25)]
    
    print('##############################')
    print('PC:')
    obj=ua.analysis(calVec)
    print('original '+str(obj.peak_compton2(coll_tot,peak['full-energy'],compton,'original')))
    print('sup '+str(obj.peak_compton2(collSup,peak['full-energy'],compton,'sup')))
    print('unsup '+str(obj.peak_compton2(collUnsup,peak['full-energy'],compton,'unsup')))
    print('peaks3 '+str(obj.peak_compton2(coll_tot.subset('n_peaks_2der',cutmin=3),peak['full-energy'],compton,'peaks3')))
    print('peaks4 '+str(obj.peak_compton2(coll_tot.subset('n_peaks_2der',cutmin=4),peak['full-energy'],compton,'peaks4')))
    print('simm '+str(obj.peak_compton2(coll_tot.subset('simm',cutmin=4),peak['full-energy'],compton,'simm')))
    print('ae '+str(obj.peak_compton2(coll_tot.subset('ae',0,ae_cut),peak['full-energy'],compton,'ae')))
    print('mix2 '+str(obj.peak_compton2(coll_tot.subset('ae',0,ae_cut).subset('n_peaks_2der',cutmin=2),peak['full-energy'],compton,'mix2')))
    print('mix3 '+str(obj.peak_compton2(coll_tot.subset('ae',0,ae_cut).subset('n_peaks_2der',cutmin=3),peak['full-energy'],compton,'mix3')))
    print('mix36 '+str(obj.peak_compton2(coll_tot.subset('ae',0,ae_cut).subset('n_peaks_2der',3,6),peak['full-energy'],compton,'mix36')))
    
    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVecR, histtype='step',label='Original')
    #c_sup, e_sup, p_sup = plt.hist(collSup.get_energies(), bins=calVecR, histtype='step',label='Supervised', color='r')
    #c_unsup, e_unsup, p_unsup = plt.hist(collUnsup.get_energies(), bins=calVecR, histtype='step',label='UNSupervised')
    #c_n, e_n, p_n = plt.hist(coll_tot.subset('n_peaks_2der',cutmin=3).get_energies(), bins=calVecR, histtype='step', label='Spettro N')
    #c_s, e_s, p_s = plt.hist(coll_tot.subset('simm',cutmin=4).get_energies(), bins=calVecR, histtype='step', label='Spettro simm')
    c_ae, e_ae, p_ae = plt.hist(coll_tot.subset('ae',0,1.90e-2).get_energies(), bins=calVecR, histtype='step', label='AE',color='black')
    c_m, e_m, p_m = plt.hist(coll_tot.subset('ae',0,1.90e-2).subset('n_peaks_2der',cutmin=3,cutmax=6).get_energies(), bins=calVecR, histtype='step', label='MIX')
       
    plt.legend(loc=2) if crioconite else plt.legend()
    plt.semilogy()
    plt.xlim(Emin,Emax)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    title = 'Crioconite' if crioconite else 'Torio'
    plt.title(title)
    plt.show()
    
    
    #AE
    plt.figure()
    Th=np.load('/home/marco/work/tesi/BEGepro/scripts/AETh.npy')
    Cr=np.load('/home/marco/work/tesi/BEGepro/scripts/AECr.npy')
    
    plt.plot(Th[1][10:],Th[0][10:],label='Th')
    plt.plot(Cr[1][10:],Cr[0][10:],label='Cr')
    
    idx_Th=Th[1][10:][np.where(Th[0][10:]==np.amax(Th[0][10:]))[0]][0]
    idx_Cr=Cr[1][10:][np.where(Cr[0][10:]==np.amax(Cr[0][10:]))[0]][0]
    plt.scatter(idx_Th,np.amax(Th[0][10:]))
    plt.scatter(idx_Cr,np.amax(Cr[0][10:]))
    
    plt.xlabel('AE')
    plt.ylabel('Picco Compton')
    plt.legend()
    plt.show()
    
    print(idx_Th)
    print(idx_Cr)
    
    
    """
    ae=coll_tot.get_avse()
    ae_max=max(ae)
    ae_min=min(ae)
    l=list([])
    l2=list([])
    for i in np.linspace(ae_min,ae_max,100):
        l.append(obj.peak_compton2(coll_tot.subset('ae',0,i),peak['full-energy'],compton,'try')[0])
        l2.append(i)
        
    plt.figure()
    plt.plot(l2,l)
    plt.show()
    
    np.save('/home/marco/work/tesi/BEGepro/scripts/AECr.npy',np.array((l,l2)))
    """    
    
    
    
    return
            

if __name__ == '__main__':
    main()
    
