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
from begepro.dspro import models as mod
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():

    # Preliminary useful variables
    crioconite = False

    a=-0.090383
    b=0.20574
    calVec = [a + b*i for i in range(2**14+1)]
    if(crioconite):
        Emax=700
        Emin=300
    else:
        Emax=2700
        Emin=1550

    # Check if GPU is available
    # train_on_gpu = torch.cuda.is_available()
    # if not train_on_gpu:
    #     print('CUDA is not available.  Training on CPU ...')
    # else:
    #     print('CUDA is available!  Training on GPU ...')
    # device = torch.device("cuda:0" if train_on_gpu else "cpu")
    # print(device)
    device = torch.device("cpu")

    # Load data

    for i in range(0,26): #36 for Th
        if(crioconite):
            dir = '/home/marco/work/Data/Analized/Crioconite/'
            filename ='CRC01-Zebru-im22012021_1_'
        else:
            dir = '/home/marco/work/Data/Analized/Torio/'
            filename='Analized228Th-grafico-tesi-im260421_1_'
        coll=ca.NPYreader(dir, filename, i, include_trace = True, include_curr = False).get_event()
        if(crioconite):
            coll.calibrate((a,b))
        if(i==0):
            coll_tot=coll.subset('energy',Emin,Emax)
        else:
            coll_tot=coll_tot+coll.subset('energy',Emin,Emax)
        del(coll)
            
        print('opened '+str(i)+' , Ram: '+str(psutil.virtual_memory()[2]))
        
    coll_tot=coll_tot.subset('energy',Emin,Emax).remove_zeros()

    # SSE: 1150-1450, 2150-2550
    # MSE: 2608-2617
    coll_MSE = coll_tot.subset('energy',2612,2618)+coll_tot.subset('energy',2101,2107) + coll_tot.subset('energy',1618,1624)
    coll_MSE = coll_MSE.subset('ae',0,1.6e-2)
    coll_SSE = coll_tot.subset('energy',1590,1596) + coll_tot.subset('energy',2250,2375)
    coll_SSE = coll_SSE.subset('ae',1.90e-2,1.95e-2)
    # index = random.sample(range(coll_MSE.n_trace),coll_SSE.n_trace)
    # coll_MSE = coll_MSE.subset('ae', index = index)
    index = random.sample(range(coll_SSE.n_trace),coll_MSE.n_trace)
    coll_SSE = coll_SSE.subset('ae', index = index) 
    print(coll_MSE.n_trace)

    print(coll_SSE.n_trace)

    coll_tot = coll_SSE + coll_MSE

    # Splitting

    indexes = range(coll_tot.n_trace)
    train, val_test = train_test_split(indexes, train_size=0.7)
    test, val = train_test_split(val_test, train_size=0.9)

    coll_train = coll_tot.subset('ae', index = train)
    coll_test = coll_tot.subset('ae', index = test)
    coll_val = coll_tot.subset('ae', index = val)

    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step',label='Original')
    c_or, e_or, p_or = plt.hist(coll_train.get_energies(), bins=calVec, histtype='step',label='Train')
    plt.semilogy()
    plt.show()


    # Create dataset and dataloader
    #data = torch.Tensor(coll_train.get_traces())
    #labels = torch.Tensor(labels)
    dataset_train = mod.Dataset(coll_train)
    loader_train = DataLoader(dataset_train, batch_size=500, shuffle=True)

    dataset_val = mod.Dataset(coll_val)
    loader_val = DataLoader(dataset_val, batch_size=100, shuffle=False)

    dataset_test = mod.Dataset(coll_test)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    
    #Check if data are loaded correctly

    # plt.figure()
    # for i, data in enumerate(loader_train, 0):
    #     inputs, labels = data
    #     print(labels)
    #     plt.plot(inputs[0].numpy())
    #     plt.show()

    # CNN
    net = mod.Conv()
    net = net.float()
    #net.to(device)

    
    filename = '/home/marco/work/Data/Models/CNN_model_best.pth'
    train_loss_list = []
    val_loss_list = []
    best_val_loss = np.inf
    epoch_best_val_loss = np.inf
    val_loss = 0
    train_loss = 0
    n_epochs = 20
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        val_loss = 0
        net.train()
        for i, data in enumerate(loader_train, 0):
            #if i>2: break
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            inputs = torch.unsqueeze(inputs, dim=1)
            # print('***',labels.shape)
            # print('***',inputs.shape)
            #labels = labels.to(device)

            # step
            labels = torch.squeeze(labels).long()
            # print('***',labels.shape)
            loss, _ = net.train_step(inputs.float(), labels)
            #if epoch==19: print(out)

            # print statistics
            train_loss += loss
        train_loss_list.append(loss)
        print('epoch ',epoch,' loss: ', loss)

        net.eval()
        with torch.no_grad():
            for i, data in enumerate(loader_val, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                inputs = torch.unsqueeze(inputs, dim=1)
                labels = torch.squeeze(labels).long()
                outputs = net.forward(inputs.float())
                loss = net.criterion(outputs, labels)
                val_loss += loss
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epoch_best_val_loss = epoch
            torch.save(net.state_dict(),filename)
        
        val_loss_list.append(val_loss)

    plt.figure()
    plt.plot(train_loss_list, label='Train')
    plt.plot(val_loss_list, label='Val')
    plt.axvline(epoch_best_val_loss)
    plt.legend()
    plt.show()

    #TESTING

    pred = []
    true = []
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader_test, 0):
            inputs, labels = data
            true.append(labels.item())
            inputs = torch.unsqueeze(inputs, dim=1)
            labels = torch.squeeze(labels).long()
            outputs = net.forward(inputs.float())
            pred.append(torch.argmax(outputs))

    accuracy = accuracy_score(true, pred, normalize=True)        
    print('Accuracy ', accuracy)

if __name__ == '__main__':
    main()