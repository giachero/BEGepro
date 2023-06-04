#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

from begepro.rw import CAENhandler_new  as ca
from begepro.dspro import models as mod
from begepro.dspro import bege_event as be
import numpy as np
import psutil
import matplotlib.pyplot as plt
import random
import pickle
from begepro.dspro import utils_analysis as utils
import os
from scipy.special import softmax

def main():

    ################################
    # PRELIMINARY USEFUL VARIABLES #
    ################################

    # To decide the radioactive source
    crioconite = False

    # To decide if you want to compute the best cut over CNN output
    compute = False

    # Calibration curve to calibrate crioconites (data are in adc channels)
    a=-0.090383
    b=0.20574
    calVec = [a + b*i for i in range(2**14+1)]

    # Setting the correct range of energies of analisis

    if(crioconite):
        Emax=700
        Emin=300
    else:
        Emax=2700
        Emin=300

    #############
    # LOAD DATA #
    #############

    # I make use of the class numpy reader that reads the signals from a numpy and returns an object bege event which 
    # is basically a data collector
    # If crioconite is loaded i perform the calibration

    startFile = 40
    endFile = 50
    for i in range(startFile, endFile):
        if(crioconite):
            dir = '/home/marco/work/Data/Analized/Crioconite/'
            filename = 'AnalizedCRC01-Zebru-im22012021_'
        else:
            dir = '/home/marco/work/Data/Analized/Torio/'
            filename='Analized228Th-grafico-tesi-im260421_1_'
        coll=ca.NPYreader(dir, filename, i, include_trace = True, include_curr = False).get_event()
        if(crioconite):
            coll.calibrate((a,b))
        if(i == startFile):
            coll_tot=coll.subset('energy',Emin,Emax)
        else:
            coll_tot=coll_tot+coll.subset('energy',Emin,Emax)
        del(coll)
            
        print('opened '+str(i)+' , Ram: '+str(psutil.virtual_memory()[2]))
        
    # This istruction is necessary in order to correctly set up data. In the data collector there are some null events to to
    # a very basic prefiltering that i implemented (i discard some noisy data)
    coll_tot=coll_tot.subset('energy',Emin,Emax).remove_zeros()

    ##############
    # EVALUATION #
    ##############

    # Useful paths
    n_data = str(startFile) + '_' + str(endFile - 1)
    labels_filename = '/home/marco/work/Data/Labels/CNN_'
    if(crioconite):
        labels_filename = labels_filename + 'Crioconite'
    else:
        labels_filename = labels_filename + '228Th'

    labels_filename = labels_filename + '_E' + str(Emin) + '_' + str(Emax)
    labels_filename = labels_filename + 'Data_' + n_data + '.npy'

    # Check if the labels are already present otherwise compute them
    if os.path.isfile(labels_filename):
        print('File detected, loading Labels')
        with open(labels_filename, 'rb') as f:
            pred = np.load(labels_filename)
        print('Loaded')
    else:
        # Load the model
        model_filename = '/home/marco/work/Data/Models/CNN_model_best.pth'
        print('File detected, loading Model')
        with open(model_filename, 'rb') as f:
            checkpoint = torch.load(model_filename, map_location = 'cpu')
            net = mod.Conv().float()
            net.load_state_dict(checkpoint)
        print('Loaded')
        net.eval()

        # Creation of dataset and dataloader
        dataset_test = mod.Dataset(coll_tot)
        loader_test = DataLoader(dataset_test, batch_size=1000, shuffle=False)

        # Evaluation of the events
        print('Start of evaluation...')
        pred = []
        with torch.no_grad():
            for i, data in enumerate(loader_test, 0):
                outputs = []
                inputs, _ = data
                inputs = torch.unsqueeze(inputs, dim=1)
                outputs = net.forward(inputs.float()).numpy()
                outputs = [softmax(outputs[i]) for i in range(outputs.shape[0])]
                for elem in outputs:
                    pred.append(elem[0])
        print('End of evaluation...')
        
        # Save predicted labels
        pred = np.array(pred)
        np.save(labels_filename, pred)

    # Set the predicted labels
    coll_tot.set_labels(pred)

    # Show them
    plt.figure()
    plt.hist(coll_tot.get_labels(),color='b',alpha=1,bins=np.arange(0,1.1,0.005),density=True)
    plt.xlabel('Label')
    plt.ylabel('Normalized counts')
    plt.show()

    ############
    # BEST CUT #
    ############

    # Generalization of the model to the whole spectre. The idea is to find the best cut over the output of CNN in order to obtain
    # the higher Peak / Compton as possible

    # Some parameters for 228Th
    comparison_energy=(2100,2110)
    peak=  {'double_escape' :(1590,1600),
            'peakBi'        :(1620,1625),         
            'full-energy'        :(2605,2620)}
    compton=(1850,2000)

    obj = utils.analysis(calVec)
    
    if compute:
        print('Start computing best cut')
        peak_compton_list = []
        cuts = []
        for i, cut in enumerate(np.linspace(0.01, 1, 100)):
            peak_compton_list.append(obj.peak_compton2(coll_tot.subset('labels', 0, cut), peak['full-energy'], compton)[0])
            cuts.append(cut)
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

    plt.figure()
    ae_cut=0.01805600127213223
    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='original')
    c_CNN, e_CNN, p_CNN = plt.hist(coll_tot.subset('labels', 0, cut).get_energies(), bins=calVec, histtype='step', label='Cut CNN')
    #c_ae, e_ae, p_ae = plt.hist(coll_tot.subset('ae',0,ae_cut).get_energies(), bins=calVec, histtype='step', label='Cut ae')
    plt.xlim(Emin, Emax)
    plt.legend()
    plt.semilogy()
    plt.show()

    # Save the energies
    filename = '/home/marco/work/Data/SavedEvents/CNN_'
    if(crioconite):
        filename = filename + 'Crioconite'
    else:
        filename = filename + '228Th'

    filename = filename + '_E' + str(Emin) + '_' + str(Emax)
    filename = filename + 'Data_' + n_data + '.npy'
    np.save(filename, coll_tot.subset('labels', 0, cut).get_energies())


if __name__ == '__main__':
    main()