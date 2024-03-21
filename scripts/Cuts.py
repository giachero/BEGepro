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
from scipy.optimize import curve_fit
from matplotlib.text import Text


import psutil
import random

import math

import pickle

import IPython 
from begepro.dspro import histfit as hf
from begepro.dspro import utils_analysis as ua

crioconite=True

def gausConst(x, ngaus, mu, sigma, p0):
    return gaus(x, ngaus, mu, sigma) + p0

def gaus(x, ngaus, mu, sigma):
    gaus = ngaus / np.sqrt(2. * np.pi * sigma ** 2) * np.exp(- 0.5 * ((x - mu) / (sigma)) ** 2)
    return gaus

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
    calVecTry = [i for i in range(2**14+1)]

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

        # Useful stuff
        live_time = 377493 * 0.71 #255172 * 0.71
        eff_br = {'666': 0.0318,
                  '609': 0.0152,
                  '480': 0.0053,
                  '354': 0.0247}
        
        # Gaussian initialization of parameters

        global models_dic, pars_dic, xlim_dic

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
                       'p0'   : 1000}
        pars_609_ACM =  {'ngaus': 10e3,
                       'mu'   : 609,
                       'sigma': 2,            
                       'p0'   : 1000}
        pars_609_GMM =  {'ngaus': 10e3,
                       'mu'   : 609,
                       'sigma': 2,            
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
        
        pars_351_or =  {'ngaus': 8000,
                       'mu'   : 354,
                       'sigma': 0.3,                       
                       'p0'   : 600}
        pars_351_ACM =  {'ngaus': 10e4,
                       'mu'   : 351,
                       'sigma': 5,                       
                       'p0'   : 1000}
        pars_351_GMM =  {'ngaus': 10e4,
                       'mu'   : 354,
                       'sigma': 0.5,                       
                       'p0'   : 1000}

        # Dictionaries containing all the dictionaries
        models_dic = {'models_666_or': ('step','const'),
                    'models_666_ACM': ('step','const'),
                    'models_666_GMM': ('step','const'),
                    'models_609_or': ('gaus','const'),
                    'models_609_ACM': ('gaus','const'),
                    'models_609_GMM': ('gaus','const'),
                    'models_480_or': ('gaus','const'),
                    'models_480_ACM': ('step','const'),
                    'models_480_GMM': ('step','const'),
                    'models_351_or': ('gaus','const'),
                    'models_351_ACM': ('gaus','const'),
                    'models_351_GMM': ('gaus','const')
                    }
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
                    'xlim_351': (340, 370)
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
    n = coll_tot.n_trace
    if crioconite: coll_tot = coll_tot.subset('ae', index=range(1500000))
    print('Events Original: ', coll_tot.n_trace)
    print('Percentage keeped: ', coll_tot.n_trace / n)

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
    if rebin: calVecR = calVecR[range(0, calVecR.shape[0], 3)]

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
    print('COUNTS CALIBRATED:', np.sum(c_or))
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
    fits = list()

    peak_or_351, counts_or_351 = fit(c_or, e_or, 351, 'or', fits, plot = True, plot_components=False)
    print(peak_or_351.get_parameters())
    print(peak_or_351.get_parameters()['sigma']['opt_value'])
    print(peak_or_351.get_parameters().keys())
    peak_auto_351, counts_auto_351 = fit(c_auto, e_auto, 351, 'ACM', fits, plot = False, plot_components=False)
    peak_GMM_351, counts_GMM_351 = fit(c_GMM, e_GMM, 351, 'GMM', fits, plot = False, plot_components=False)

    fits.append(None)
    peak_auto_480, counts_auto_480 = fit(c_auto, e_auto, 480, 'ACM', fits, plot = False, plot_components=False)
    peak_GMM_480, counts_GMM_480 = fit(c_GMM, e_GMM, 480, 'GMM', fits, plot = False, plot_components=False)

    peak_or_609, counts_or_609 = fit(c_or, e_or, 609, 'or', fits, plot = False, plot_components=False)
    print('\n')
    print(peak_or_609.get_parameters())
    peak_auto_609, counts_auto_609 = fit(c_auto, e_auto, 609, 'ACM', fits, plot = False, plot_components=False)
    peak_GMM_609, counts_GMM_609 = fit(c_GMM, e_GMM, 609, 'GMM', fits, plot = False, plot_components=False)

    peak_or_666, counts_or_666 = fit(c_or, e_or, 666, 'or', fits, plot = False, plot_components=False)
    peak_auto_666, counts_auto_666 = fit(c_auto, e_auto, 666, 'ACM', fits, plot = False, plot_components=False)
    peak_GMM_666, counts_GMM_666 = fit(c_GMM, e_GMM, 666, 'GMM', fits, plot = False, plot_components=False)

    # Check fits parameters
    print('CHECK FITS')
    print('peak_or_351 ', check_fit(peak_or_351))
    print('peak_auto_351 ', check_fit(peak_auto_351))
    print('peak_GMM_351 ', check_fit(peak_GMM_351))
    print('peak_auto_480 ', check_fit(peak_auto_480))
    print('peak_GMM_480 ', check_fit(peak_GMM_480))
    print('peak_or_609 ', check_fit(peak_or_609))
    print('peak_auto_609 ', check_fit(peak_auto_609))
    print('peak_GMM_609 ', check_fit(peak_GMM_609))
    print('peak_or_666 ', check_fit(peak_or_666))
    print('peak_auto_666 ', check_fit(peak_auto_666))
    print('peak_GMM_666 ', check_fit(peak_GMM_666))

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
    frac = evaluate_fraction(counts_GMM_666, counts_or_666)
    print('GMM / Original: %f +- %f'%(frac[0], frac[1]))
    
    print('\n')
    print('Compton reduction:')
    print('Auto / Original: %f'%(compton_counts(e_auto, c_auto, compton) / compton_counts(e_or, c_or, compton)))
    print('GMM / Original: %f'%(compton_counts(e_GMM, c_GMM, compton) / compton_counts(e_or, c_or, compton)))

    # bin_spacing = e_or[1] - e_or[0]

    activities = {'351_or'   : activity(counts_or_351, 25.7e-3, eff_br['354'], live_time),
                  '609_or'   : activity(counts_or_609, 25.7e-3, eff_br['609'], live_time),
                  '666_or'   : activity(counts_or_666, 25.7e-3, eff_br['666'], live_time),
                  '351_auto' : activity(counts_auto_351, 25.7e-3, eff_br['354'], live_time),
                  '609_auto' : activity(counts_auto_609, 25.7e-3, eff_br['609'], live_time),
                  '666_auto' : activity(counts_auto_666, 25.7e-3, eff_br['666'], live_time),
                  '351_GMM'  : activity(counts_GMM_351, 25.7e-3, eff_br['354'], live_time),
                  '609_GMM'  : activity(counts_GMM_609, 25.7e-3, eff_br['609'], live_time),
                  '666_GMM'  : activity(counts_GMM_666, 25.7e-3, eff_br['666'], live_time)
                }
    
    print('\n')
    print('Activities')
    for key in activities.keys():
        print('Activity {}: {}'.format(key, activities[key]))

    ratios = {'351_or_vs_auto' : evaluate_fraction(counts_or_351, counts_auto_351),
              '609_or_vs_auto' : evaluate_fraction(counts_or_609, counts_auto_609),
              '666_or_vs_auto' : evaluate_fraction(counts_or_666, counts_auto_666),
              '351_or_vs_GMM'  : evaluate_fraction(counts_or_351, counts_GMM_351),
              '609_or_vs_GMM'  : evaluate_fraction(counts_or_609, counts_GMM_609),
              '666_or_vs_GMM'  : evaluate_fraction(counts_or_666, counts_GMM_666)
            }
    
    print('\n')
    print('Ratios')
    for key in ratios.keys():
        print('Difference {}: {}'.format(key, ratios[key]))

    # Verify that activities are consistent
    print('\n')
    print('Activity consistency')
    print('ACM')
    print("351 auto ", activities['351_auto'][0] * ratios['351_or_vs_auto'][0], " vs ", activities['351_or'][0])
    print("609 auto ", activities['609_auto'][0] * ratios['609_or_vs_auto'][0], " vs ", activities['609_or'][0])
    print("666 auto ", activities['666_auto'][0] * ratios['666_or_vs_auto'][0], " vs ", activities['666_or'][0])
    print('GMM')
    print("351 GMM ", activities['351_GMM'][0] * ratios['351_or_vs_GMM'][0], " vs ", activities['351_or'][0])
    print("609 GMM ", activities['609_GMM'][0] * ratios['609_or_vs_GMM'][0], " vs ", activities['609_or'][0])
    print("666 GMM ", activities['666_GMM'][0] * ratios['666_or_vs_GMM'][0], " vs ", activities['666_or'][0])

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
    # plt.xlim(450, 510)
    # plt.ylim(10e2, 10e3)
    plt.legend(loc = 1)
    plt.ylabel('Counts / 1.23 KeV', fontsize=fs)
    title = 'Cryoconite' if crioconite else '228Th'
    # plt.title(title)
    plt.show()

    # Scatter ratios counts

    plt.figure()
    x = [351, 609, 661]
    y_ACM = [ratios['351_or_vs_auto'][0], ratios['609_or_vs_auto'][0], ratios['666_or_vs_auto'][0]]
    err_ACM =  [ratios['351_or_vs_auto'][1], ratios['609_or_vs_auto'][1], ratios['666_or_vs_auto'][1]]
    y_GMM = [ratios['351_or_vs_GMM'][0], ratios['609_or_vs_GMM'][0], ratios['666_or_vs_GMM'][0]]
    err_GMM =  [ratios['351_or_vs_GMM'][1], ratios['609_or_vs_GMM'][1], ratios['666_or_vs_GMM'][1]]
    plt.scatter(x, y_ACM, color = 'r', label='ACM')
    plt.scatter(x, y_GMM, color = 'orange', label='GMM')
    plt.errorbar(x, y_ACM, err_ACM, color = 'r', fmt="o")
    plt.errorbar(x, y_GMM, err_GMM, color = 'orange', fmt="o")

    fit_ACM, cov = curve_fit(linearFunc, x, y_ACM, sigma=err_ACM, absolute_sigma=True)
    inter_ACM = fit_ACM[0]
    slope_ACM = fit_ACM[1]
    inter_err_ACM = np.sqrt(cov[0][0])
    slope_err_ACM = np.sqrt(cov[1][1])

    fit_GMM, cov = curve_fit(linearFunc, x, y_GMM, sigma=err_GMM, absolute_sigma=True)
    inter_GMM = fit_GMM[0]
    slope_GMM = fit_GMM[1]
    inter_err_GMM = np.sqrt(cov[0][0])
    slope_err_GMM = np.sqrt(cov[1][1])

    yfit_ACM = [inter_ACM + slope_ACM * i for i in x]
    yfit_GMM = [inter_GMM + slope_GMM * i for i in x]
    plt.plot(x, yfit_ACM, color = 'black')
    plt.plot(x, yfit_GMM, color = 'black')
    plt.xlabel('Energy [KeV]')
    plt.ylabel('Counts ratio')
    plt.legend()
    plt.show()

    # print('Reduced chi^2 ACM = ', chisq(x, yfit_ACM, inter_ACM, slope_ACM, err_y))
    # print('Reduced chi^2 GMM = ', chisq(x, yfit_GMM, inter_GMM, slope_GMM, err_y))

    factor_ACM = InterpolatelinearFunc(peak_auto_480.get_parameters()['mu']['opt_value'], inter_ACM, slope_ACM, inter_err_ACM, slope_err_ACM)
    factor_GMM = InterpolatelinearFunc(peak_auto_480.get_parameters()['mu']['opt_value'], inter_GMM, slope_GMM, inter_err_GMM, slope_err_GMM)

    print("Factor of Be for ACM is", factor_ACM)
    print("Activity Be for ACM", activity(counts_auto_480, 25.7e-3, eff_br['480'], live_time, factor = factor_ACM))
    print("Factor of Be for GMM is", factor_GMM)
    print("Activity Be for GMM", activity(counts_auto_480, 25.7e-3, eff_br['480'], live_time, factor = factor_GMM))

    # Grid of fits

    fig, axs = plt.subplots(3,4)
    plt.subplots_adjust(wspace = 0.265, hspace=0.1)

    fits = np.array(fits).reshape(4,3)
    lw = 2
    fs = 12

    for j in range(4): 
        axs[0][j].hist(coll_tot.get_energies(), bins = calVecR, histtype='step', label = 'Original', linewidth=lw)
        if j != 1: 
            fits[:, 0][j].plot_fit(plot = axs[0][j], lw = lw)

    for j in range(4): 
        axs[1][j].hist(collAutoencoder.get_energies(), bins = calVecR, histtype = 'step', label = 'ACM', color = 'r', linewidth=lw)
        fits[:, 1][j].plot_fit(plot = axs[1][j], lw = lw)
    for j in range(4): 
        axs[2][j].hist(collGMM.get_energies(), bins = calVecR, histtype = 'step', label = 'GMM', color = 'orange', linewidth=lw)
        fits[:, 2][j].plot_fit(plot = axs[2][j], lw = lw)

    row_labels = ('Original', 'ACM', 'GMM')
    col_labels = ('214Pb', '7Be', '214Bi', '137Cs')
    for j, ax in enumerate(axs[0, :]):
        ax.annotate(col_labels[j], xy=(0.5, 1), xytext=(0, 10),
                    xycoords='axes fraction', textcoords='offset points',
                    ha='center', va='center', fontsize=fs)

    # Add labels for rows

    for i, ax in enumerate(axs[:, 0]):
        ax.annotate(row_labels[i], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad -0 if i==0 else -17, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    ha='center', va='center', rotation=90, fontsize=fs)
    
    for i in axs:
        for ax in i:
            ax.tick_params(axis='both', which = 'both', labelsize=fs)
            # ax.tick_params(axis='both', which = 'minor', labelsize=10)
            
    for a in axs[:,0]: 
        a.set_xlim(peak_or_351.get_parameters()['mu']['opt_value']-10, peak_or_351.get_parameters()['mu']['opt_value']+10)
        a.set_ylim(1.1e2, 1e4)

    for i,a in enumerate(axs[:,1]): 
        a.set_xlim(470, 490)
        if i != 0: a.set_ylim([5e2, 1e3])

    for a in axs[:,2]: 
        a.set_xlim(peak_or_609.get_parameters()['mu']['opt_value']-10, peak_or_609.get_parameters()['mu']['opt_value']+10)
        a.set_ylim(5e1, 4e3)

    for a in axs[:,3]: 
        a.set_xlim(peak_or_666.get_parameters()['mu']['opt_value']-10, peak_or_666.get_parameters()['mu']['opt_value']+10)

    for i in range(3):
        for j in range(4):
            axs[i][j].semilogy()
            if i != 2: axs[i][j].set_xticklabels([])
    axs[0][0].set_ylim(1e3,8e3)
    old_labels = axs[0][0].get_yticklabels()
    axs[0][0].set_yticklabels([Text(0, 10.0, '$\\mathdefault{10^{1}}$'), Text(0, 100.0, ''), Text(0, 1000.0, ''), Text(0, 10000.0, ''), Text(0, 100000.0, '$\\mathdefault{10^{5}}$')])

    plt.show()

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

def fit(c_or, e_or, energy, model, fits, plot = False, plot_components = False):
    s_models = 'models_' + str(energy) + '_' + model
    s_lim = 'xlim_' + str(energy)
    s_pars = 'pars_' + str(energy) + '_' + model
    peak = hf.HistogramFitter(c_or, e_or, e_or[1]-e_or[0])
    peak.set_model(models_dic[s_models], xlim = xlim_dic[s_lim], initpars = pars_dic[s_pars])
    peak.fit()
    if plot: peak.plot_fit()
    if plot_components: peak.plot_components()
    fits.append(peak)
    return peak, peak.net_counts()

def check_fit(pars):
    pars = pars.get_parameters()
    for k in pars.keys():
        if pars[k]['opt_value'] < 0:
            # print('value ', pars[k]['opt_value'])
            return False
    return True

def activity(counts, m, br_times_eff, t, factor = None):
    if factor is not None:
        err_counts = np.sqrt((counts[1] * factor[0])**2 + (factor[1] * counts[0])**2)
        counts = (counts[0] * factor[0], err_counts)
    act = counts[0] / (m * br_times_eff * t)
    err = np.sqrt((counts[1] / (m * br_times_eff * t))**2 + (act * 0.05)**2)
    return act, err

def diff_activities(activity_one, activity_two):
    diff = activity_one[0] - activity_two[0]
    err = np.sqrt(activity_one[1] * activity_one[1] + activity_two[1] * activity_two[1])
    return diff, err

def linearFunc(x, intercept, slope):
    y = intercept + slope * x
    return y

def chisq(x, y, inter, slope, err_y):
    chisqr = 0
    for i in range(len(x)):
        chisqr += (y[i] - linearFunc(x[i], inter, slope))**2 / err_y[i]**2
    dof = len(y) - 2
    return chisqr / dof

def InterpolatelinearFunc(x, intercept, slope, err_intercept, err_slope):
    y = linearFunc(x, intercept, slope)
    err = np.sqrt(err_intercept ** 2 + (err_slope * x)**2)
    return y, err

            

if __name__ == '__main__':
    main()
    