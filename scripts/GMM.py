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


import psutil
import random

import math

import pickle

import IPython 
from begepro.dspro import histfit as hf
from begepro.dspro import utils_analysis as ua

calVec = [-0.090383 + 0.20574*i for i in range(2**14+1)]
crioconite=False
train=False
fit=False

def main():

    #OPEN FILES

    print('\n')
    print('\n')
    if(crioconite):
        Emax=700
        #Emax=2650
        Emin=100
        #Emin=1550
    else:
        Emax=2650
        Emin=1550
        Emin=150
    
    
    for i in range(5,45,5): #45 for crioconite
        if(crioconite):
            filename='/home/marco/work/tesi/data/NewParameters/Crioconite/NoTrace/CRC01-Zebru-im22012021__'+str(i)+'.npy'
        else:
            filename='/home/marco/work/tesi/data/NewParameters/228Th/NoTrace/228Th-grafico-tesi-im260421_1__'+str(i)+'.npy'
        coll=ca.NPYreader(filename,False).get_event()
        if(crioconite):
            coll.calibrate((-0.09,0.205))
        if(i==5):
            coll_tot=coll.subset('energy',Emin,Emax)
        else:
            coll_tot=coll_tot+coll.subset('energy',Emin,Emax)
            del(coll)
            
        print('opened '+str(i)+' , Ram: '+str(psutil.virtual_memory()[2]))
        
    coll_tot=coll_tot.subset('energy',Emin,Emax)
        
    matrix=coll_tot.get_parameters()
      
    plt.figure()
    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='Spettro originario')
    plt.semilogy()
    plt.xlim(Emin,Emax)
    plt.show()
    
    
    #DATA STRUCTURE

    df=pd.DataFrame(matrix,columns=['index']+list(coll_tot.get_dict().keys())[2:-2])
    df['index']=np.arange(0,len(df['index']))
    df_par=df[['risetime','simm','area','n_peaks_2der','zeros_2der','ae']]
    df_par=(df_par-df_par.mean())/df_par.std()
    print(df_par)
    del(df)
    
    #NAMING STUFF
    
    clusters=3
    n_data=0
    name='/home/marco/work/tesi/data/GMM_'
    if(crioconite):
        name=name+'Crioconite'
    else:
        name=name+'228Th'
    name=name+'_c'+str(clusters)+'_e'+str(Emin)+'_'
    n_data='_'+str(n_data)+'.npy'
    
    #GMM
    X=np.array(df_par)
    if(train):
        model=mixture.GaussianMixture(n_components=clusters, covariance_type="diag")
        X=np.array(df_par)
        model.fit(X)

        labels=model.predict(X)
        #centers=model.centers
        
        #np.save(name+'centres'+n_data,centers)
        np.save(name+'labelsTot'+n_data,labels)
        
        file_name = name+'model'+n_data[0:(len(n_data)-3)]+'pkl'
        
        with open(file_name,'wb') as file:
            pickle.dump(model,file)
    
    
    #SET LABELS
    if(fit):
        if(crioconite):
            file_name ='/home/marco/work/tesi/data/GMM_Crioconite_c2_e1550_model_0.pkl'
        else:
            file_name ='/home/marco/work/tesi/data/GMM_228Th_c2_e1550_model_0.pkl'
        
        X=np.array(df_par)
        filehandler = open(file_name, 'rb') 
        model = pickle.load(filehandler)
        labels=model.predict(X)
    else:    
        labels=np.load(name+'labelsTot'+n_data)
    
    coll_tot.set_labels(labels)
    print('\n')
    print(labels)
    
    #HIST OF LABELS VS AE
    
    plt.figure()
    for i in range(0,clusters):
        c_or, e_or, p_or = plt.hist(coll_tot.subset('labels',i,i+1).get_avse(),bins=np.linspace(0,0.05,500),histtype='step',density=True, alpha=0.7,label='Label '+str(i))
  
    
    plt.xlabel('ae')
    plt.ylabel('Conteggi normalizzati')
    plt.legend()
    plt.show()
    
    if(bool(input())):
        labels=np.where(labels==1,np.zeros(len(labels)),np.ones(len(labels)))
        coll_tot.set_labels(labels)
    #coll_tot=coll_tot.subset('labels',0,0.5)
        
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
                'peakTl'        :(2605,2620)}
        compton=(1850,2000)    
    
    #SPECTRUM
    obj=ua.analysis(calVec)
    #obj.histogram(coll_tot,1.5,comparison_energy,(Emin,Emax))
    
    plt.figure()
    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='Spettro originario')
    c_AI, e_AI, p_AI = plt.hist(coll_tot.subset('labels',0.5).get_energies(), bins=calVec, histtype='step', label='Spettro AI')
    
    c_counts=np.max(c_AI[np.where((e_AI>comparison_energy[0]) & (e_AI<comparison_energy[1]))[0]])
    ae=coll_tot.get_avse()
    ae_max=max(ae)
    ae_min=min(ae) 
    cut=obj.findCutAE(ae_min,ae_max,c_counts,coll_tot,comparison_energy)
    c_ae, e_ae, p_ae = plt.hist(coll_tot.subset('ae',0,cut[0]).get_energies(), bins=calVec, histtype='step', label='Spettro AE')
    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='Spettro originario')
    c_AI, e_AI, p_AI = plt.hist(coll_tot.subset('labels',0.5).get_energies(), bins=calVec, histtype='step', label='Spettro AI')
    plt.semilogy()
    plt.grid(axis='x')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    plt.xlim(Emin,Emax)
    plt.legend(loc='upper left')
    plt.semilogy()
    plt.show()
    
    
    return
            

if __name__ == '__main__':
    main()
    
