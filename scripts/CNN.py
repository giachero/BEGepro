#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

from begepro.rw import CAENhandler_new  as ca
import numpy as np
import psutil
import matplotlib.pyplot as plt
import random
import pickle
import os
from begepro.dspro import models as mod
from begepro.dspro import utils_analysis as utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from sklearn.metrics import auc
from sklearn import metrics

def main():

    ################################
    # PRELIMINARY USEFUL VARIABLES #
    ################################

    # To decide the radioactive source
    crioconite = False

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
        Emin=1550

    #############
    # LOAD DATA #
    #############

    # I make use of the class numpy reader that reads the signals from a numpy and returns an object bege event which 
    # is basically a data collector
    # If crioconite is loaded i perform the calibration

    for i in range(0,40): #36 for Th
        if(crioconite):
            dir = '/home/marco/work/Data/Analized/Crioconite/'
            filename = 'CRC01-Zebru-im22012021_1_'
        else:
            dir = '/home/marco/work/Data/Analized/Torio/'
            filename = 'Analized228Th-grafico-tesi-im260421_1_'
        coll = ca.NPYreader(dir, filename, i, include_trace = True, include_curr = False).get_event()
        if(crioconite):
            coll.calibrate((a,b))
        if(i == 0):
            coll_tot = coll.subset('energy', Emin, Emax)
        else:
            coll_tot = coll_tot+coll.subset('energy', Emin, Emax)
        del(coll)
            
        print('opened '+str(i)+' , Ram: '+str(psutil.virtual_memory()[2]))
        
    # This istruction is necessary in order to correctly set up data. In the data collector there are some null events to to
    # a very basic prefiltering that i implemented (i discard some noisy data)
    coll_tot = coll_tot.subset('energy',Emin,Emax).remove_zeros()

    # Spectrum in order to see the loaded data  
    plt.figure()
    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='Spettro originario')
    plt.semilogy()
    plt.xlim(Emin,Emax)
    plt.title('Spectrum of ' + ('Cryoconite' if crioconite else '228Th'))
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    plt.show()

    #################################
    # TRAINING, VALIDATION AND TEST #
    #################################

    # Here using the collector method subset i can select the events i want to keep. The first argument select the parameter
    # on which i want to perform the cut and the following two it's minimum and maximum.
    # I loved also implemented the overload of the operator +
    # The cuts on energy and ae are performed according to teoretical values in order to isolate single and multi site events
    
    coll_MSE = coll_tot.subset('energy',2450,2618)+coll_tot.subset('energy',2101,2107) + coll_tot.subset('energy',1618,1624) + coll_tot.subset('energy',1618,1624)
    coll_MSE = coll_MSE.subset('ae',0,1.80e-2)
    coll_SSE = coll_tot.subset('energy',1590,1596) + coll_tot.subset('energy',2250,2375)
    coll_SSE = coll_SSE.subset('ae',cutmin=1.90e-2)

    # Since MSE are much more than SSE i artificially balance the dataset randomly extracting events. Due to a bad implementation
    # of the method subset it always needs a string as first argument, but if index is passed the previous arguments are ignored

    index = random.sample(range(coll_MSE.n_trace),coll_SSE.n_trace)
    coll_MSE = coll_MSE.subset('ae', index = index)
    # index = random.sample(range(coll_SSE.n_trace),coll_MSE.n_trace)
    # coll_SSE = coll_SSE.subset('ae', index = index)

    # Check if the dataset is balances
    print(coll_MSE.n_trace)
    print(coll_SSE.n_trace)
     # Visualize the regions where the data was taken
    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step',label='Original')
    c_or, e_or, p_or = plt.hist(coll_MSE.get_energies(), bins=calVec, histtype='step',label='MSE', color = 'green')
    c_or, e_or, p_or = plt.hist(coll_SSE.get_energies(), bins=calVec, histtype='step',label='SSE', color = 'red')
    plt.xlim(Emin,Emax)
    plt.legend()
    plt.semilogy()
    plt.show()

    # Considered data
    coll_tot = coll_SSE + coll_MSE

    # Actual splitting in train (70%), validation (27%) and test (3%). 

    indexes = range(coll_tot.n_trace)
    train, val_test = train_test_split(indexes, train_size=0.7)
    test, val = train_test_split(val_test, train_size=0.9)

    coll_train = coll_tot.subset('ae', index = train)
    coll_test = coll_tot.subset('ae', index = test)
    coll_val = coll_tot.subset('ae', index = val)

    # Create dataset and dataloader useful to operate with torch library

    dataset_train = mod.Dataset(coll_train)
    loader_train = DataLoader(dataset_train, batch_size=500, shuffle=True)

    dataset_val = mod.Dataset(coll_val)
    loader_val = DataLoader(dataset_val, batch_size=100, shuffle=False)

    dataset_test = mod.Dataset(coll_test)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    ###########
    # NETWORK #
    ###########

    # CNN
    net = mod.Conv()
    net = net.float()
    
    # Useful variables. The idea is to save the best model according to the best validation loss found
    train_loss_list = []
    val_loss_list = []
    best_val_loss = np.inf
    epoch_best_val_loss = np.inf
    val_loss = 0
    train_loss = 0
    filename = '/home/marco/work/Data/Models/CNN_model_best.pth'

    # 100 epochs. A plateau is reached
    n_epochs = 100

    # if model is already saved, load it else perform the training

    if os.path.isfile(filename):
        print('File detected, loading ...')
        with open(filename, 'rb') as f:
            checkpoint = torch.load(filename, map_location = 'cpu')
            net = mod.Conv().float()
            net.load_state_dict(checkpoint)
        print('Loaded')
    else:

        # Training and validation loop
        print('Start training...')

        for epoch in range(n_epochs):
            train_loss = 0.0
            val_loss = 0

            # Training

            net.train()
            for i, data in enumerate(loader_train, 0):
                # get the inputs
                inputs, labels = data
                inputs = torch.unsqueeze(inputs, dim=1)
                labels = torch.squeeze(labels).long()

                loss, _ = net.train_step(inputs.float(), labels)

                train_loss += loss
            train_loss_list.append(loss)
            print('epoch ',epoch,' loss: ', loss)

            # Validation

            net.eval()
            with torch.no_grad():
                for i, data in enumerate(loader_val, 0):
                    inputs, labels = data
                    inputs = torch.unsqueeze(inputs, dim=1)
                    labels = torch.squeeze(labels).long()
                    outputs = net.forward(inputs.float())
                    loss = net.criterion(outputs, labels)
                    val_loss += loss
            
            # Save the model if is the best found

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epoch_best_val_loss = epoch
                torch.save(net.state_dict(),filename)
            
            val_loss_list.append(val_loss)

        print('End of training')
        
        # Loss plots

        plt.figure()
        plt.plot(range(n_epochs),train_loss_list, label='Train')
        plt.plot(range(n_epochs),val_loss_list, label='Val')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.axvline(epoch_best_val_loss, color = 'black', linestyle=(0, (5, 5)))
        plt.legend()
        plt.show()

    ###########
    # TESTING #
    ###########

    pred = []
    true = []
    pred_classes = []
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader_test, 0):
            inputs, labels = data
            true.append(labels.item())
            inputs = torch.unsqueeze(inputs, dim=1)
            labels = torch.squeeze(labels).long()
            outputs = net.forward(inputs.float()).numpy()
            outputs = softmax(outputs[0])
            pred_classes.append(np.argmax(outputs))
            pred.append(outputs[1])

    accuracy = accuracy_score(true, pred_classes, normalize=True)        
    print('Accuracy 0.5 threshold ', accuracy)

    fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
    print(metrics.auc(fpr, tpr))

    plt.figure()
    plt.plot(fpr,tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()

if __name__ == '__main__':
    main()