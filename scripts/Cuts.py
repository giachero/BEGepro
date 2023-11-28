#!/usr/bin/env python

from begepro.rw import CAENhandler_new  as ca
from begepro.dspro import histfit as hf
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
crioconite=True

def main():

    ################################
    # PRELIMINARY USEFUL VARIABLES #
    ################################

    # To decide models to use
    CNN = False

    # To decide the radioactive source
    crioconite = True

    # To decide if you want to compute the best cut over GMM output
    compute = True

    # To decide if rebin the histograms or not
    rebin = True

    # Calibration curve to calibrate crioconites (data are in adc channels)
    a=-0.090383
    b=0.20574
    calVec = [a + b*i for i in range(2**14+1)]

    # Setting the correct range of energies of analisis

    if(crioconite):
        Emax = 700
        Emin = 300
    else:
        Emax = 2700
        Emin = 300

    # Setting some parameters for some gaussian fits

    if(crioconite):
        #comparison_energy=(605,615)

        # Energies of the photopeak and it's compton
        peak = {'full-energy' :(660,670)}
        compton = (355,450)
        
        # Best ae cut
        ae_cut = 0.01441559398334617
        
        # Gaussian initialization of parameters

        global shape, bkg, pars_dic, xlim_dic

        shape = 'step'
        bkg = 'const'

        # 666 keV

        pars_666_or =  {'ngaus': 10e6,
                       'mu'   : 666,
                       'sigma': 5,           
                       'cstep': 500,            
                       'p0'   : 1000}
        pars_666_ACM =  {'ngaus': 10e6,
                       'mu'   : 666,
                       'sigma': 5,           
                       'cstep': 500,            
                       'p0'   : 1000}
        pars_666_GMM =  {'ngaus': 10e6,
                       'mu'   : 666,
                       'sigma': 5,           
                       'cstep': 500,            
                       'p0'   : 1000}
        
        # 609 keV
        
        pars_609_or =  {'ngaus': 10e3,
                       'mu'   : 609,
                       'sigma': 2,           
                       'cstep': 100,            
                       'p0'   : 1000}
        pars_609_ACM =  {'ngaus': 10e3,
                       'mu'   : 609,
                       'sigma': 2,           
                       'cstep': 100,            
                       'p0'   : 1000}
        pars_609_GMM =  {'ngaus': 10e3,
                       'mu'   : 609,
                       'sigma': 2,           
                       'cstep': 100,            
                       'p0'   : 1000}
        
        # 480 keV
        
        pars_480_or =  {'ngaus': 10e3,
                       'mu'   : 480,
                       'sigma': 2,           
                       'cstep': 100,            
                       'p0'   : 1000}
        pars_480_ACM =  {'ngaus': 10e2,
                       'mu'   : 480,
                       'sigma': 2,           
                       'cstep': 100,            
                       'p0'   : 1000}
        pars_480_GMM =  {'ngaus': 10e2,
                       'mu'   : 480,
                       'sigma': 2,           
                       'cstep': 100,            
                       'p0'   : 1000}
        
        # 351 keV
        
        pars_351_or =  {'ngaus': 10e4,
                       'mu'   : 351,
                       'sigma': 5,           
                       'cstep': 500,            
                       'p0'   : 1000}
        pars_351_ACM =  {'ngaus': 10e4,
                       'mu'   : 351,
                       'sigma': 5,           
                       'cstep': 500,            
                       'p0'   : 1000}
        pars_351_GMM =  {'ngaus': 10e4,
                       'mu'   : 351,
                       'sigma': 5,           
                       'cstep': 500,            
                       'p0'   : 1000}

        # Dictionaries containing all the dictionaries         
        pars_dic = {'pars_666_or': pars_666_or,
                    'pars_666_ACM': pars_666_ACM,
                    'pars_666_GMM': pars_666_GMM,
                    'pars_609_or': pars_609_or,
                    'pars_609_ACM': pars_609_ACM,
                    'pars_609_GMM': pars_609_GMM,
                    'pars_480_or': pars_480_or,
                    'pars_480_ACM': pars_480_ACM,
                    'pars_480_GMM': pars_480_GMM,
                    'pars_351_or': pars_351_or,
                    'pars_351_ACM': pars_351_ACM,
                    'pars_351_GMM': pars_351_GMM
                    } 
        xlim_dic = {'xlim_666': (636, 696),
                    'xlim_609': (600, 630),
                    'xlim_480': (465, 495),
                    'xlim_351': (345, 360)
                    }
        
    else:
        # comparison_energy=(2100,2110)
        # comparison_energy=(580,590)

        # Energies of the photopeaks and the full energy compton
        peak=  {'double_escape' :(1590,1600),
                'peakBi'        :(1620,1625),         
                'full-energy'        :(2605,2620)}
        compton=(1850,2000)
        
        # Best ae cut
        ae_cut=0.01805600127213223
        
        # Gaussian initialization of parameters
        pars_Tl =  {'ngaus': 6500,
                    'mu'   : 2615,
                    'sigma': 2,
                    'ntail': 10**2,
                    'ttail': 2.5,
                    'cstep': 50,            
                    'p0'   : 10**1}
        
        pars_fe =  {'ngaus': 100,
                    'mu'   : 2104,
                    'sigma': 1,           
                    'p0'   : 10}
        
        pars_Bi =  {'ngaus': 100,
                    'mu'   : 1621,
                    'sigma': 1,           
                    'p0'   : 10}
        
        pars_de =  {'ngaus': 1750,
                    'mu'   : 1593,
                    'sigma': 2,           
                    'p0'   : 10**1}

        # Dictionaries containing the other ones                
        peak_dic = {'pars_Tl': pars_Tl, 'pars_fe': pars_fe, 'pars_Bi': pars_Bi, 'pars_de': pars_de}
        xlim_dic = {'pars_Tl': (2580,2625),'pars_fe': (2094,2114),'pars_Bi': (1611,1631),'pars_de': (1583,1603)}

    #############
    # LOAD DATA #
    #############

    # I make use of the class numpy reader that reads the signals from a numpy and returns an object bege event which 
    # is basically a data collector
    # If crioconite is loaded i perform the calibration
    startFile = 0
    endFile = 29
    for i in range(startFile, endFile):
        if(crioconite):
            dir = '/home/marco/work/Data/Analized/Crioconite/'
            filename = 'AnalizedCRC01-Zebru-im22012021_'
        else:
            dir = '/home/marco/work/Data/Analized/Torio/'
            filename = 'Analized228Th-grafico-tesi-im260421_1_'
        coll = ca.NPYreader(dir, filename, i, include_trace = False, include_curr = False).get_event()
        if(crioconite):
            coll.calibrate((a,b))
        if(i == startFile):
            coll_tot = coll.subset('energy', Emin, Emax)
        else:
            coll_tot = coll_tot+coll.subset('energy', Emin, Emax)
        del(coll)
            
        print('opened '+str(i)+' , Ram: '+str(psutil.virtual_memory()[2]))
        
    # This istruction is necessary in order to correctly set up data. In the data collector there are some null events to to
    # a very basic prefiltering that i implemented (i discard some noisy data)
    coll_tot = coll_tot.subset('energy',Emin,Emax).remove_zeros()
    if crioconite: coll_tot = coll_tot.subset('ae', index=range(1500000))
    print('Events Original: ', coll_tot.n_trace)

    # Loading as bege events several saved energies of different AI models

    # AUTOENCODER
    filename = '/home/marco/work/Data/SavedEvents/selected_adc_channels_Cr.npy' if crioconite else '/home/marco/work/Data/SavedEvents/selected_adc_channels_Th.npy'
    comp = np.load(filename)
    
    collAutoencoder = be.BEGeEvent(len(comp), coll_tot.dim_trace, pheight = comp)
    collAutoencoder.calibrate((a,b))
    print('Events Autoencoder: ', collAutoencoder.n_trace)

    # CNN
    if(CNN):
        for i in range(40, 70, 10):
            if(crioconite):
                dir = '/home/marco/work/Data/Analized/Crioconite/'
                filename = 'CNN_228Cr_E300_700Data_'+str(i) + '_' + str(i + 9) + '.npy'
            else:
                dir = '/home/marco/work/Data/SavedEvents/'
                filename = 'CNN_228Th_E300_2700Data_'+str(i) + '_' + str(i + 9) + '.npy'
                print(filename)
            energies = np.load(dir + filename)
            coll = be.BEGeEvent(len(energies), coll_tot.dim_trace, energy = energies)
            if i == 40:
                collCNN = coll
            else:
                collCNN += coll
            del(coll)

        print('Events CNN: ', collCNN.n_trace)
    
    # GMM 
    filename = '/home/marco/work/Data/SavedEvents/GMM_energies_Cr_.npy' if crioconite else '/home/marco/work/Data/SavedEvents/GMM_energies.npy'
    energies = np.load(filename)
    
    collGMM = be.BEGeEvent(len(energies), coll_tot.dim_trace, energy = energies)
    print('Events GMM: ', collGMM.n_trace)
    
    
    #############################
    # PEAK / COMPTON EVALUATION #
    #############################

    # Necessary to rebin the histogram and enhance the background reduction 
    calVecR = np.array(calVec)
    if rebin: calVecR = calVecR[range(0, calVecR.shape[0], 6)]

    # Object in which there is the method peak_compton2 which doesn't make use of fits, but evaluates it as
    # counts of the higher bin of the peak / summation of counts in the bins the constitutes the Compton
    obj = ua.analysis(calVec)
    
    print('#### Peak / Compton ####')
    print('Original '+str(obj.peak_compton2(coll_tot, peak['full-energy'], compton)))
    print('Autoencoder '+str(obj.peak_compton2(collAutoencoder, peak['full-energy'], compton)))
    # print('CNN '+str(obj.peak_compton2(collCNN, peak['full-energy'],compton)))
    print('GMM '+str(obj.peak_compton2(collGMM, peak['full-energy'], compton)))
    print('AE '+str(obj.peak_compton2(coll_tot.subset('ae', 0, ae_cut), peak['full-energy'], compton)))

    ae_best = 0.014940998585613832
    # coll_mix = coll_tot.subset('n_peaks_2der',cutmin=3).subset('simm',cutmin=3).subset('risetime', cutmin = 0.3e-6).subset('ae', 0, ae_cut)
    # print('MIX '+str(obj.peak_compton2(coll_mix, peak['full-energy'], compton)))

    # OTHER EXPERIMENTAL CUTS
    # ignore them

    # print('peaks3 '+str(obj.peak_compton2(coll_tot.subset('n_peaks_2der',cutmin=3),peak['full-energy'],compton)))
    # print('peaks4 '+str(obj.peak_compton2(coll_tot.subset('n_peaks_2der',cutmin=4),peak['full-energy'],compton)))
    # print('simm '+str(obj.peak_compton2(coll_tot.subset('simm',cutmin=4),peak['full-energy'],compton,'simm')))
    # print('mix2 '+str(obj.peak_compton2(coll_tot.subset('ae',0,ae_cut).subset('n_peaks_2der',cutmin=2),peak['full-energy'],compton,'mix2')))
    # print('mix3 '+str(obj.peak_compton2(coll_tot.subset('ae',0,ae_cut).subset('n_peaks_2der',cutmin=3),peak['full-energy'],compton,'mix3')))
    # print('mix36 '+str(obj.peak_compton2(coll_tot.subset('ae',0,ae_cut).subset('n_peaks_2der',3,6),peak['full-energy'],compton,'mix36')))
    
    plt.figure()
    plt.rcParams['axes.linewidth'] = 2
    lw = 2
    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins = calVecR, histtype='step', label = 'Original', linewidth=lw)
    c_auto, e_auto, p_auto = plt.hist(collAutoencoder.get_energies(), bins = calVecR, histtype = 'step', label = 'ACM', color = 'r', linewidth=lw)
    # c_CNN, e_CNN, p_CNN = plt.hist(collCNN.get_energies(), bins = calVecR, histtype = 'step', label = 'CNN', linewidth=lw)
    c_GMM, e_GMM, p_GMM = plt.hist(collGMM.get_energies(), bins = calVecR, histtype = 'step', label = 'GMM', linewidth=lw)
    # c_ae, e_ae, p_ae = plt.hist(coll_tot.subset('ae', 0, ae_cut).get_energies(), bins = calVecR, histtype = 'step', label = 'AE',color = 'black', linewidth=lw)

    # c_n, e_n, p_n = plt.hist(coll_tot.subset('n_peaks_2der',cutmin=4).get_energies(), bins=calVecR, histtype='step', label='Spettro N4')

    #########################
    # QUANTITATIVE MEASURES #
    #########################

    # FITS

    original_max = np.max(c_or)
    auto_max = np.max(c_auto)
    GMM_max = np.max(c_GMM)

    counts_or_666 = fit(c_or, e_or, 666, 'or', plot = False)
    counts_auto_666 = fit(c_auto, e_auto, 666, 'ACM', plot = False)
    counts_GMM_666 = fit(c_GMM, e_GMM, 666, 'GMM', plot = False)

    counts_or_609 = fit(c_or, e_or, 609, 'or', plot = False)
    counts_auto_609 = fit(c_auto, e_auto, 609, 'ACM', plot = False)
    counts_GMM_609 = fit(c_GMM, e_GMM, 609, 'GMM', plot = False)

    # peak = hf.HistogramFitter(c_auto,e_auto)
    # peak.set_model(('step','no'), xlim = xlim_dic['xlim_480'], initpars = pars_dic['pars_480_ACM'])
    # peak.fit()
    # print(peak.get_parameters())
    # peak.plot_fit()
    # peak.plot_components()

    counts_auto_480 = fit(c_auto, e_auto, 480, 'ACM', plot = True)
    counts_GMM_480 = fit(c_GMM, e_GMM, 480, 'GMM', plot = True)

    counts_or_351 = fit(c_or, e_or, 351, 'or', plot = False)
    counts_auto_351 = fit(c_auto, e_auto, 351, 'ACM', plot = False)
    counts_GMM_351 = fit(c_GMM, e_GMM, 351, 'GMM', plot = False)

    print('\n')
    print(' Counts original 666 keV:', counts_or_666)
    print(' Counts ACM 666 keV:', counts_auto_666)
    print(' Counts GMM 666 keV:', counts_GMM_666)

    print('\n')
    print(' Counts original 609 keV:', counts_or_609)
    print(' Counts ACM 609 keV:', counts_auto_609)
    print(' Counts GMM 609 keV:', counts_GMM_609)

    print('\n')
    # print(' Counts original 480 keV:', counts_or_480)
    print(' Counts ACM 480 keV:', counts_auto_480)
    print(' Counts GMM 480 keV:', counts_GMM_480)

    print('\n')
    print(' Counts original 351 keV:', counts_or_351)
    print(' Counts ACM 351 keV:', counts_auto_351)
    print(' Counts GMM 351 keV:', counts_GMM_351)

    print('\n')
    print('Peak reduction bin:')
    print('Auto / Original: %f'%(auto_max / original_max))
    print('GMM / Original: %f'%(GMM_max / original_max))

    print('\n')
    print('Peak reduction fit:')
    frac = evaluate_fraction(counts_auto_666, counts_or_666)
    print('Auto / Original: %f +- %f'%(frac[0], frac[1]))
    frac = evaluate_fraction(counts_GMM_666,counts_or_666)
    print('GMM / Original: %f +- %f'%(frac[0], frac[1]))
    
    print('\n')
    print('Compton reduction:')
    print('Auto / Original: %f'%(compton_counts(e_auto, c_auto, compton) / compton_counts(e_or, c_or, compton)))
    print('GMM / Original: %f'%(compton_counts(e_GMM, c_GMM, compton) / compton_counts(e_or, c_or, compton)))

    # print('Auto / Original: '+str(evaluate_fraction(counts_auto,counts_or_666)[0])+' +- '+str(evaluate_fraction(counts_auto,counts_or_666)[1]))
    # print('GMM / Original: '+str(evaluate_fraction(counts_GMM,counts_or_666)[0])+' +- '+str(evaluate_fraction(counts_GMM,counts_or_666)[1]))

    # print(counts_auto[0])
    # print(counts_GMM[0])
    # print(counts_or_666[0])

    # OTHER EXPERIMENTAL CUTS
    # ignore them
    # c_mix, e_mix, p_mix = plt.hist(coll_mix.get_energies(), bins=calVecR, histtype='step', label='Mix')
    #c_s, e_s, p_s = plt.hist(coll_tot.subset('simm',cutmin=4).get_energies(), bins=calVecR, histtype='step', label='Spettro simm')
    #c_m, e_m, p_m = plt.hist(coll_tot.subset('ae',0,1.90e-2).subset('n_peaks_2der',cutmin=3,cutmax=6).get_energies(), bins=calVecR, histtype='step', label='MIX')

    # Location of the legend

    fs = 18  
    plt.legend(loc=2, fontsize=fs) if crioconite else plt.legend(fontsize=fs)
    plt.semilogy()
    plt.xlim(Emin + 3,Emax)

    plt.tick_params(axis='both', labelsize=fs)

    plt.xlabel('Energy [keV]', fontsize=fs)
    plt.xlim(450, 510)
    plt.ylim(10e2, 10e3)
    plt.legend(loc = 1)
    plt.ylabel('Counts / 1.23 KeV', fontsize=fs)
    title = 'Cryoconite' if crioconite else '228Th'
    # plt.title(title)
    plt.show()

    print(calVecR[1]-calVecR[0])

    ######################
    # IGNORE FROM NOW ON #
    ######################

    
    # print('AE:')
    # for i in peak_dic.keys():
    #     res = obj.counts(coll_tot.subset('ae',0,ae_cut), coll_tot,xlim_dic[i],peak_dic[i], double = i in ['pars_Tl','pars_pp'])
    #     print(i, ' :', res[0], ' +- ', res[1])
        
    # #print('SUP:')
    # #for i in peak_dic.keys():
    #     #res = obj.counts(collSup, coll_tot,xlim_dic[i],peak_dic[i],double = i in ['pars_Tl','pars_pp'])
    #     #print(i, ' :', res[0], ' +- ', res[1])
        
    # print('UNSUP:')
    # for i in peak_dic.keys():
    #     res = obj.counts(collUnsup, coll_tot,xlim_dic[i],peak_dic[i],double = i in ['pars_Tl','pars_pp'])
    #     print(i, ' :', res[0], ' +- ', res[1])
        
    # print('MIX:')
    # for i in peak_dic.keys():
    #     res = obj.counts(coll_tot.subset('ae',0,ae_cut).subset('n_peaks_2der',3,6), coll_tot,xlim_dic[i],peak_dic[i],double = i in ['pars_Tl','pars_pp'])
    #     print(i, ' :', res[0], ' +- ', res[1])
    
    
    # #AE
    # plt.figure()
    # Th=np.load('/home/marco/work/tesi/BEGepro/scripts/AETh.npy')
    # Cr=np.load('/home/marco/work/tesi/BEGepro/scripts/AECr.npy')
    
    # plt.plot(Th[1][10:],Th[0][10:],label='Th')
    # plt.plot(Cr[1][10:],Cr[0][10:],label='Cr')
    
    # idx_Th=Th[1][10:][np.where(Th[0][10:]==np.amax(Th[0][10:]))[0]][0]
    # idx_Cr=Cr[1][10:][np.where(Cr[0][10:]==np.amax(Cr[0][10:]))[0]][0]
    # plt.scatter(idx_Th,np.amax(Th[0][10:]))
    # plt.scatter(idx_Cr,np.amax(Cr[0][10:]))
    
    # plt.xlabel('AE')
    # plt.ylabel('Picco Compton')
    # plt.legend()
    # plt.show()
    
    # print(idx_Th)
    # print(idx_Cr)
    
    
    
    # #Best AE
    # """
    # ae=coll_tot.get_avse()
    # ae_max=max(ae)
    # ae_min=min(ae)
    # l=list([])
    # l2=list([])
    # for i in np.linspace(ae_min,ae_max,100):
    #     l.append(obj.peak_compton2(coll_tot.subset('ae',0,i),peak['full-energy'],compton,'try')[0])
    #     l2.append(i)
        
    # plt.figure()
    # plt.plot(l2,l)
    # plt.show()
    
    # np.save('/home/marco/work/tesi/BEGepro/scripts/AECr.npy',np.array((l,l2)))
    # """    
 
    return

def evaluate_fraction(a,b):
    r=a[0]/b[0]
    err=math.sqrt((a[1]/b[0])**2+(a[0]*b[1]/(b[0])**2)**2)
    return (r,err)

def compton_counts(e_hist, c_hist, compton):
    bounds = np.where((e_hist>compton[0]) & (e_hist<compton[1]))
    return np.sum(c_hist[bounds])

def fit(c_or, e_or, energy, model, plot = False, plot_components = False):
    s_lim = 'xlim_' + str(energy)
    s_pars = 'pars_' + str(energy) + '_' + model
    peak = hf.HistogramFitter(c_or,e_or)
    peak.set_model((shape,bkg), xlim = xlim_dic[s_lim], initpars = pars_dic[s_pars])
    peak.fit()
    if plot: peak.plot_fit()
    if plot_components: peak.plot_components()
    
    return peak.net_counts()
            

if __name__ == '__main__':
    main()
    