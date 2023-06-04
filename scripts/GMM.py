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

import os


import psutil
import random

import math

import pickle

import IPython 
from begepro.dspro import histfit as hf
from begepro.dspro import utils_analysis as utils

def main():

    ################################
    # PRELIMINARY USEFUL VARIABLES #
    ################################

    # To decide the radioactive source
    crioconite = False

    # To decide if you want to compute the best cut over GMM output
    compute = False

    # To decide if fit or train the model
    train = False
    fit = False

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

    #############
    # LOAD DATA #
    #############

    # I make use of the class numpy reader that reads the signals from a numpy and returns an object bege event which 
    # is basically a data collector
    # If crioconite is loaded i perform the calibration
    startFile = 40
    endFile = 70
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

    # Get the parameters    
    matrix=coll_tot.get_parameters()

    # Spectrum in order to see the loaded data  
    plt.figure()
    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='Spettro originario')
    plt.semilogy()
    plt.xlim(Emin,Emax)
    plt.title('Spectrum of ' + ('Cryoconite' if crioconite else '228Th'))
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    plt.show()
    
    # Before applying clustering i select the desired variables and i scale them.
    # Risetime, simm are continuous while n_peaks_2der is not. Applying the scaler in such a manner 
    # is not properly correct. 
    ##### TO BE FIXED BEFORE ARTICLE #####
    # Since has to be changed may use scaler from sklearn

    df=pd.DataFrame(matrix, columns=['index'] + list(coll_tot.get_dict().keys())[2:-2])
    df['index'] = np.arange(0, len(df['index']))
    df_par = df[['simm','n_peaks_2der','risetime']]
    df_par = (df_par-df_par.mean()) / df_par.std()
    print(df_par)
    del(df)
    X = np.array(df_par)

    ###########################
    # GAUSSIAN MIXTURE MODELS #
    ###########################
    
    # Useful variables

    # Useful variable to try fitting different times and change the name of the model
    n_data = str(startFile) + '_' + str(endFile - 1)

    #Construction of the name of the file
    filename = '/home/marco/work/Data/Labels/GMM_'
    if(crioconite):
        filename = filename + 'Crioconite'
    else:
        filename = filename + '228Th'

    filename = filename + '_E' + str(Emin) + '_' + str(Emax)
    filename = filename + 'Data_' + n_data + '.npy'

    # Useful paths
    labels_filename = filename
    model_filename = '/home/marco/work/Data/Models/GMM_model_' 
    model_filename += 'Cr_' if crioconite else 'Th'
    model_filename += '.pkl'
    
    # Construction of the model
    
    if os.path.isfile(labels_filename):
        # The labels are alredy present
        print('Found labels. Loading them ...')

        labels = np.load(labels_filename)

        print('Loaded')
        
    elif os.path.isfile(model_filename):
        # The model exists
        print('Found model. Loading it ...')
        print(model_filename)

        # Open
        filehandler = open(model_filename, 'rb') 
        model = pickle.load(filehandler)
        print('Loaded')

        # Predict classes
        print('Predicting classes ...')
        labels = model.predict_proba(X)
        print('Done')

        # Save them
        np.save(labels_filename, labels)  
        print('Saved')

    else: 
        # Train the model
        print('Fitting data ...')
        model = mixture.GaussianMixture(n_components = 2, covariance_type = "full")
        model.fit(X)
        labels = model.predict_proba(X)
        print('Done')

        # Save model and labels
        np.save(labels_filename, labels)  

        with open(model_filename,'wb') as file:
            pickle.dump(model,file)

        print('Saved')
    
    # Set labels
    coll_tot.set_labels(labels[:, 0])
    print(coll_tot.get_labels())

    # Show them
    plt.figure()
    plt.hist(coll_tot.get_labels(),color='b',alpha=1,bins=np.arange(0,1.1,0.005),density=True)
    plt.xlabel('Label')
    plt.ylabel('Normalized counts')
    plt.show()

    ############
    # BEST CUT #
    ############

    # Some parameters for 228Th and cryoconite
    if(crioconite):
        #Crioconite
        comparison_energy=(605,615)
        peak={  'full-energy' :(660,670)}
        compton=(355,395)
    else:
        #228Th
        comparison_energy=(2100,2110)
        peak={  'double_escape' :(1590,1600),
                'peakBi'        :(1620,1625),         
                'full-energy'   :(2605,2620)}
        compton=(1850,2000)

    obj = utils.analysis(calVec)
    
    if compute:
        print('Start computing best cut')
        peak_compton_list = []
        cuts = []
        for i, cut in enumerate(np.linspace(0.01, 1, 100)):
            peak_compton_list.append(obj.peak_compton2(coll_tot.subset('labels', 0, cut), peak['full-energy'], compton)[0])
            cuts.append(i)
            if i//10 == 0: print((i+1) * 10,'%')

        fig = plt.figure()
        plt.scatter([round(elem, 2) for elem in cuts], peak_compton_list)
        plt.xticks(rotation = 90)
        plt.xlabel('Cut')
        plt.ylabel('Peak / Compton')
        plt.show()

        cut = np.linspace(0.01, 1, 100)[np.argmax(peak_compton_list)]
        print('Computed ', cut)
    else:
        cut = 0.1

    # I want to save the MSE as class 0. So I cycle until done
    while(True):
        # Invert labels
        coll_tot.set_labels(1 - coll_tot.get_labels())
        plt.figure()
        c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='original')
        c_CNN, e_CNN, p_CNN = plt.hist(coll_tot.subset('labels', 0, cut).get_energies(), bins=calVec, histtype='step', label='Cut GMM')
        plt.xlim(Emin, Emax)
        plt.legend()
        plt.semilogy()
        plt.show()
        if input("Are MSE class 0? "): break
    
    # Save the energies
    filename = '/home/marco/work/Data/SavedEvents/GMM_energies.npy'
    np.save(filename, coll_tot.subset('labels', 0, cut).get_energies())


    ####################
    # IGNORE THIS PART #
    ####################
        
    # #HIST OF LABELS VS AE
    
    # plt.figure()
    # for i in range(0,clusters):
    #     coll_tot.set_labels(labels[:,i])
    #     c_or, e_or, p_or = plt.hist(coll_tot.subset('labels',0.8,1).get_avse(),bins=np.linspace(0,0.05,500),histtype='step',density=True, alpha=0.7,label='Label '+str(i))
    
    # plt.xlabel('ae')
    # plt.ylabel('Conteggi normalizzati')
    # plt.legend()
    # plt.show()
    # j=int(input())
    # coll_tot.set_labels(1-labels[:,j])
    
    # #HIST OF LABELS
    # plt.figure()
    # plt.hist(coll_tot.get_labels(),color='b',alpha=1,bins=np.arange(0,1.1,0.005),density=True)
    # plt.xlabel('Label')
    # plt.ylabel('Conteggi normalizzati')
    # plt.show()
 
    # #HIST 2D
    
    # fig,axs=plt.subplots(3,figsize=(20,20))
    # for i in range(0,clusters):
    #     r=axs[i].hist2d(coll_tot.get_avse(),labels[:,i], bins=(np.linspace(0.01,0.03,600),np.linspace(0,1,500)), cmap=plt.cm.turbo, norm=clr.LogNorm())
    #     axs[i].set(xlabel='ae',ylabel='label '+str(i))
    # plt.show()
    
    
    # #SCATTER GRID PLOT
    
    # n=len(df_par.keys())
    # maxP=50000
    # keys=df_par.keys()
    # fig,axs=plt.subplots(n,n,figsize=(40,20))
    # for i in range(0,n):
    #     for j in range(0,n):
    #         r=axs[i][j].hist2d(df_par[keys[i]][0:maxP],df_par[keys[j]][0:maxP], bins=(np.linspace(-2,2,200),np.linspace(-2,2,200)), cmap=plt.cm.turbo, norm=clr.LogNorm())
    #         axs[i][j].set(xlabel=keys[i],ylabel=keys[j])
            
    # fig.subplots_adjust(hspace=0.5)
    # fig.subplots_adjust(wspace=0.3)
    # plt.show()
   
    # #CMvsAE EXPERIMENTAL
    # labels2=np.array(1-labels[:,2])
    # labels0=np.array(labels[:,0])
    # #z=np.zeros((1,len(labels0)))
    
    # labels=np.where((labels0>0.6) & (labels0<1),labels0,labels2)
    # print(labels)
    # coll_tot.set_labels(labels)

    
    
    
    # #CMvsAE
    # obj=ua.analysis(calVec)
    # k=np.linspace(0.1,1,10)
    # obj.AIvsAE(coll_tot,peak,comparison_energy,compton,k)
        
    
    # #SPECTRUM
    # cutAI=float(input())
    # obj.histogram(coll_tot,cutAI,comparison_energy,(Emin,Emax))
    
    
    return
            

if __name__ == '__main__':
    main()
    
