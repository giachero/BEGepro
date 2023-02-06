#!/usr/bin/env python

from begepro.rw import CAENhandler_new  as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split 

from fcmeans import FCM 

import psutil
import random

import math

import IPython 
from begepro.dspro import histfit as hf

calVec = [-0.090383 + 0.20574*i for i in range(2**14+1)]

def main():

    #OPEN FILES

    print('\n')
    print('\n')
    Emax=2650
    Emin=1550   #1550
    for i in range(55,260,5): #1550: 260 ,300: 155, 150: 105
        filename='/home/marco/work/tesi/data/228Th-grafico-tesi-im260421_1/AnalisiBig3/228Th-grafico-tesi-im260421_1__'+str(i)+'.npy'
        coll=ca.NPYreader(filename,False).get_event()

        if(i==55):
            coll_tot=coll.subset('energy',Emin,Emax)
        else:
            coll_tot=coll_tot+coll.subset('energy',Emin,Emax)
            del(coll)
            
        print('opened '+str(i)+' , Ram: '+str(psutil.virtual_memory()[2]))
        
    coll_tot=coll_tot.subset('energy',Emin,Emax)
    matrix=coll_tot.get_parameters()
    
    #DATA STRUCTURE

    df=pd.DataFrame(matrix,columns=['index']+list(coll_tot.get_dict().keys())[2:-2])
    df['index']=np.arange(0,len(df['index']))
    df_par=df[['risetime','simm']]
    df_par=(df_par-df_par.mean())/df_par.std()
    print(df_par)
    del(df)
    
    m=2
    clusters=3
    n_data=1
    name='/home/marco/work/tesi/data/FCM_c'+str(clusters)+'_e'+str(Emin)+'_'
    n_data='_'+str(n_data)+'.npy'
    #FUZZY CM
    """
    model=FCM(n_clusters=clusters,m=m)
    X=np.array(df_par)
    model.fit(X)
    labels=model.soft_predict(X)
    #labels1=labels[:,0]
    #labels2=labels[:,1]
    #labels3=labels[:,2]
    centers=model.centers
    print(labels)
    print(centers)
    
    np.save(name+'centres'+n_data,centers)
    #np.save(name+'labels'+n_data,labels1)
    np.save(name+'labelsTot'+n_data,labels)
    """
    #SET LABELS
    
    centers=np.load(name+'centres'+n_data)
    labels=np.load(name+'labelsTot'+n_data)
    #labels1=labels[:,0]
    #labels2=labels[:,1]
    #labels3=labels[:,2]
    #print(labels1)
    #labels1=1-labels1
    #np.save(name+'labels'+n_data,labels1)
    #coll_tot.set_labels(labels1)
    print('\n')
    
    #HIST OF LABELS VS AE
    
    plt.figure()
    for i in range(0,clusters):
        coll_tot.set_labels(labels[:,i])
        c_or, e_or, p_or = plt.hist(coll_tot.subset('labels',0.8,1).get_avse(),bins=np.linspace(0,0.05,500),histtype='step',density=True, alpha=0.7,label='Label '+str(i))
        """
    c_or, e_or, p_or = plt.hist(coll_tot.subset('labels',0.8,1).get_avse(),bins=np.linspace(0,0.05,500),histtype='step',density=True, alpha=0.7,color='b',label='Label Two')
    coll_tot.set_labels(labels3)
    c_or, e_or, p_or = plt.hist(coll_tot.subset('labels',0.8,1).get_avse(),bins=np.linspace(0,0.05,500),histtype='step',density=True, alpha=0.7,color='g',label='Label Three')
    """
    plt.legend()
    plt.show()
    j=int(input())
    coll_tot.set_labels(1-labels[:,j])
    
    #PLOT OF LABELS
    plt.figure()
    plt.hist(1-labels[:,j],color='b',alpha=1,bins=np.arange(0,2.1,0.005),density=True)
    plt.xlabel('Label')
    plt.ylabel('Conteggi normalizzati')
    plt.xticks(np.linspace(0,1.1,100))
    plt.show()
    
    #SCATTER GRID PLOT
    """
    maxP=10000
    keys=df_par.keys()
    fig,axs=plt.subplots(3,3,figsize=(10,10))
    for i in range(0,3):
        for j in range(0,3):
            axs[i][j].scatter(df_par[keys[i]][0:maxP],df_par[keys[j]][0:maxP],alpha=0.005)
            axs[i][j].set(xlabel=keys[i],ylabel=keys[j])
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df_par[keys[0]][0:maxP],df_par[keys[1]][0:maxP],df_par[keys[2]][0:maxP],alpha=0.005)
    for i in range(0,clusters):
        ax.scatter(centers[i][0],centers[i][1],centers[i][2], marker="+", s=500, c='r')
    plt.show()
    del(df_par)
    """
    #CMvsAE
    k=np.linspace(0.1,1,10)
    CMvsAE(coll_tot,k)
    
    #SPECTRUM
    cutCM=float(input())
    histogram(cutCM,coll_tot,(Emin,Emax))
    
    #HIST OF CUTS
    
    plt.figure()
    c_or, e_or, p_or = plt.hist(coll_tot.subset('labels',0.66,0.68).get_avse(),bins=np.linspace(0,0.05,500),histtype='step',density=True, alpha=0.7,color='r',label='Picco medio')
    c_or, e_or, p_or = plt.hist(coll_tot.subset('labels',0.95,1).get_avse(),bins=np.linspace(0,0.05,500),histtype='step',density=True,alpha=0.7, color='b',label='Picco fondo')
    c_or, e_or, p_or = plt.hist(coll_tot.subset('labels',0,0.5).get_avse(),bins=np.linspace(0,0.05,500),histtype='step',density=True,alpha=0.7,color='green', label='Plateau')
    c_or, e_or, p_or = plt.hist(coll_tot.subset('labels',0.7,0.9).get_avse(),bins=np.linspace(0,0.05,500),histtype='step',density=True,alpha=0.7,color='black', label='Plateau2')
    plt.legend()
    plt.show()
    
    """
    #EXP
 
    collf=coll_tot.subset('labels',0,0.66)+coll_tot.subset('labels',0.7,0.8)
    
    #CUT CM         
        
    plt.figure()
    c_NN, e_NN, p_NN =plt.hist(collf.get_energies(), bins=calVec, histtype='step', label='zeros')
        #c_ae, e_ae, p_ae =plt.hist(coll_tot.subset('ae',0,cut_ae).get_energies(), bins=calVec, histtype='step', label='ae',color='black',alpha=0.5)
        
    peak = hf.HistogramFitter(c_NN, e_NN)
    xlim = 2090,2115

    shape = 'gaus'
    bkg = 'const'


    pars = {'ngaus': 10**4,
                'mu'   : 2104,
                'sigma': 2,                
                'p0'   : 10**1}
        
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    c_counts=peak.net_counts()[0]
    peak.plot_fit()
    plt.legend()
    #plt.show()
       
    ae=coll_tot.get_avse()
    ae_max=max(ae)
    ae_min=min(ae) 
    cut=findCutAE2(ae_min,ae_max,c_counts,coll_tot)
    
    
    
    
    plt.figure()
    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='Spettro originario')
    c_CM, e_CM, p_CM = plt.hist(collf.get_energies(), bins=calVec, histtype='step', label='Spettro CM',color='black')
    c_ae, e_ae, p_ae = plt.hist(coll_tot.subset('ae',0,cut[0]).get_energies(), bins=calVec, histtype='step', label='Spettro AE')
    plt.semilogy()
    plt.grid(axis='x')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    plt.xlim(1550,2650)
    plt.legend(loc='upper left')
    plt.semilogy()
    plt.show()
    """
    
    return
    
def evaluate_fraction(a,b):
    r=a[0]/b[0]
    err=math.sqrt((a[1]/b[0])**2+(a[0]*b[1]/(b[0])**2)**2)
    return (r,err)
    
def fit_peak(xlim,pars,c,e):
    peak = hf.HistogramFitter(c,e)
    xlim = 1590,1600

    shape = 'gaus'
    bkg = 'const'
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    
    #peak.plot_fit()
    
    return peak
    
def peak_compton(c_or,e_or,c_CM,e_CM,c_ae,e_ae,peak_or,peak_CM,peak_ae):

    #PhotoPeak 208Tl/Compton
    bounds=np.where((e_or>1850) & (e_or<2000))
    original=evaluate_fraction((peak_or,0),(np.sum(c_or[bounds]),0))[0] , evaluate_fraction((peak_or,0),(np.sum(c_or[bounds]),0))[1]
    bounds=np.where((e_ae>1850) & (e_ae<2000))
    ae=evaluate_fraction((peak_ae,0),(np.sum(c_ae[bounds]),0))[0] , evaluate_fraction((peak_ae,0),(np.sum(c_ae[bounds]),0))[1]
    bounds=np.where((e_CM>1850) & (e_CM<2000))
    CM=evaluate_fraction((peak_CM,0),(np.sum(c_CM[bounds]),0))[0] , evaluate_fraction((peak_CM,0),(np.sum(c_CM[bounds]),0))[1]
    
    return {'original': original,
            'CM'   : CM,
            'ae': ae}
    
def analysis(c_or,e_or,c_CM,e_CM,c_ae,e_ae):

    #ORIGINAL
    #Double escape  
    double_escape_or=np.max(c_or[np.where((e_or>1590) & (e_or<1600))[0]])
    
    #Peak 212Bi
    peakBi_or=np.max(c_or[np.where((e_or>1620) & (e_or<1625))[0]])
    
    #Peak first escape
    first_escape_or=np.max(c_or[np.where((e_or>2100) & (e_or<2110))[0]])  
        
    #Peak 208Tl
    peakTl_or=np.max(c_or[np.where((e_or>2605) & (e_or<2620))[0]])

    
    #CM
    #Double escape
    double_escape_CM=np.max(c_CM[np.where((e_CM>1590) & (e_CM<1600))[0]])
    
    #Peak 212Bi
    peakBi_CM=np.max(c_CM[np.where((e_CM>1620) & (e_CM<1625))[0]])
    
    #Peak first escape
    first_escape_CM=np.max(c_CM[np.where((e_CM>2100) & (e_CM<2110))[0]])
        
    #Peak 208Tl
    peakTl_CM=np.max(c_CM[np.where((e_CM>2605) & (e_CM<2620))[0]])
  
        
    #AE
    #Double escape
    double_escape_ae=np.max(c_ae[np.where((e_ae>1590) & (e_ae<1600))[0]])
    
    #Peak 212Bi   
    peakBi_ae=np.max(c_ae[np.where((e_ae>1620) & (e_ae<1625))[0]])
    
    #Peak first escape
    first_escape_ae=np.max(c_ae[np.where((e_ae>2100) & (e_ae<2110))[0]])
        
    #Peak 208Tl
    peakTl_ae=np.max(c_ae[np.where((e_ae>2605) & (e_ae<2620))[0]])
    
    return {'double_escape_or': double_escape_or,
            'peakBi_or': peakBi_or,
            'first_escape_or': first_escape_or,
            'peakTl_or'   : peakTl_or,
            'double_escape_CM': double_escape_CM,
            'peakBi_CM': peakBi_CM,
            'first_escape_CM': first_escape_CM,
            'peakTl_CM'   : peakTl_CM,
            'double_escape_ae': double_escape_ae,
            'peakBi_ae': peakBi_ae,
            'first_escape_ae': first_escape_ae,
            'peakTl_ae'   : peakTl_ae}
   
    
def findCutAE(ae_min,ae_max,c_counts,coll_tot):
    cut_ae_prev=0
    for i in np.linspace(ae_min,ae_max,100):
      
        coll_ae=coll_tot.subset('ae',0,i)
        try:
            c, e, p = plt.hist(coll_ae.get_energies(), bins=calVec, histtype='step', label='Spettro tagliato')
            ae_counts=np.max(c[np.where((e>2100) & (e<2110))[0]])
        
            #print('rapporto ae: '+str(ae_counts/len(labels))+' , cut: '+str(i))
                    
            if(ae_counts>c_counts):
                cut_ae=i
                break
            else:
                cut_ae_prev=i
        except:
            print("")
        
    return cut_ae,cut_ae_prev,ae_counts
   
def findCutAE2(ae_min,ae_max,c_counts,coll_tot):
    eps=0.001
   
    while(True):
        cut=(ae_max+ae_min)/2 
        #print(cut)
        coll_ae=coll_tot.subset('ae',0,cut)
        #print(coll_ae)    
        c, e, p = plt.hist(coll_ae.get_energies(), bins=calVec, histtype='step', label='Spettro tagliato')
        ae_counts=np.max(c[np.where((e>2100) & (e<2110))[0]])
        plt.close('all')
        
        val=abs(ae_counts/c_counts-1)
        #print(val)
        if(val<eps):
            print('y')
            break
        elif(ae_counts>c_counts):
            ae_max=cut
        else:
            ae_min=cut
                    
    return cut,ae_counts
    
def CMvsAE(coll_tot,k):
    o_tl=np.array([])
    c_tl=np.array([])
    a_tl=np.array([])
    o_de=np.array([])
    c_de=np.array([])
    a_de=np.array([])
    o_bi=np.array([])
    c_bi=np.array([])
    a_bi=np.array([])
    
    for i in k:
        cutCM=i
        #cutCM=float(input())
            
        #CUT CM         
            
        plt.figure()
        c_CM, e_CM, p_CM =plt.hist(coll_tot.subset('labels',0,cutCM).get_energies(), bins=calVec, histtype='step', label='zeros')
        c_counts=np.max(c_CM[np.where((e_CM>2100) & (e_CM<2110))[0]])


        #CUT AE
            
        ae=np.sort(coll_tot.get_avse())
        ae_max=max(ae)
        ae_min=min(ae) 
        cut=findCutAE2(ae_min,ae_max,c_counts,coll_tot)
            
        print(float(cut[1])/c_counts)
            
            
        #Spectrum
        #Here i find the optimal cuts for label and ae
            

        plt.figure()
        c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='Spettro originario')
        c_CM, e_CM, p_CM = plt.hist(coll_tot.subset('labels',0,cutCM).get_energies(), bins=calVec, histtype='step', label='Spettro CM')
        c_ae, e_ae, p_ae = plt.hist(coll_tot.subset('ae',0,cut[0]).get_energies(), bins=calVec, histtype='step', label='Spettro AE')
        plt.semilogy()
        plt.grid(axis='x')
        plt.xlabel('Energy [keV]')
        plt.ylabel('Counts')
        plt.xlim(1550,2650)
        plt.legend(loc='upper left')
        plt.semilogy()
        #plt.show()
        plt.close('all')
        print('done '+str(i))
               
            
        #ANALISIS
            
        res=analysis(c_or, e_or,c_CM, e_CM,c_ae, e_ae)           
        pc=peak_compton(c_or,e_or,c_CM,e_CM,c_ae,e_ae,res['peakTl_or'],res['peakTl_CM'],res['peakTl_ae'])
        o_tl=np.append(o_tl,pc['original'][0]) 
        c_tl=np.append(c_tl,pc['CM'][0]) 
        a_tl=np.append(a_tl,pc['ae'][0]) 
        pc=peak_compton(c_or,e_or,c_CM,e_CM,c_ae,e_ae,res['double_escape_or'],res['double_escape_CM'],res['double_escape_ae'])
        o_de=np.append(o_de,pc['original'][0]) 
        c_de=np.append(c_de,pc['CM'][0]) 
        a_de=np.append(a_de,pc['ae'][0])
        pc=peak_compton(c_or,e_or,c_CM,e_CM,c_ae,e_ae,res['peakBi_or'],res['peakBi_CM'],res['peakBi_ae'])
        o_bi=np.append(o_bi,pc['original'][0]) 
        c_bi=np.append(c_bi,pc['CM'][0]) 
        a_bi=np.append(a_bi,pc['ae'][0])
            
    plt.figure()
    plt.scatter(k,o_tl,marker='o',c='red')#,label='or')
    plt.scatter(k,c_tl,marker='x',c='red')#,label='CM')
    plt.scatter(k,a_tl,marker='s',c='red')#,label='ae')
    
    plt.scatter(k,o_de,marker='o',c='blue')#,label='or')
    plt.scatter(k,c_de,marker='x',c='blue')#,label='CM')
    plt.scatter(k,a_de,marker='s',c='blue')#,label='ae')
    
    plt.scatter(k,o_bi,marker='o',c='green')#,label='or')
    plt.scatter(k,c_bi,marker='x',c='green')#,label='CM')
    plt.scatter(k,a_bi,marker='s',c='green')#,label='ae')
    
    handles=list()
    red_patch=mpatches.Patch(color='red',label='208Tl')
    blue_patch=mpatches.Patch(color='blue',label='Double escape')
    green_patch=mpatches.Patch(color='green',label='212Bi')
    o_line=mlines.Line2D([],[],color='black',marker='o',markersize=10,ls="",label='Original')
    CM_line=mlines.Line2D([],[],color='black',marker='x',markersize=10,ls="",label='CM')
    AE_line=mlines.Line2D([],[],color='black',marker='s',markersize=10,ls="",label='AE')
    handles.append(red_patch)
    handles.append(blue_patch)
    handles.append(green_patch)
    handles.append(o_line)
    handles.append(CM_line)
    handles.append(AE_line)
    plt.legend(handles=handles)
    plt.xlabel('Cut CM')
    plt.ylabel('n_peak / Compton')
    plt.show()
    
    return
    
def histogram(cutCM,coll_tot,lim):

    plt.figure()
    c_CM, e_CM, p_CM =plt.hist(coll_tot.subset('labels',0,cutCM).get_energies(), bins=calVec, histtype='step', label='zeros')
    c_counts=np.max(c_CM[np.where((e_CM>2100) & (e_CM<2110))[0]])
        
    ae=np.sort(coll_tot.get_avse())
    ae_max=max(ae)
    ae_min=min(ae) 
    cut=findCutAE2(ae_min,ae_max,c_counts,coll_tot)
    
    plt.figure()
    c_or, e_or, p_or = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='Spettro originario')
    c_CM, e_CM, p_CM = plt.hist(coll_tot.subset('labels',0,cutCM).get_energies(), bins=calVec, histtype='step', label='Spettro CM')
    c_ae, e_ae, p_ae = plt.hist(coll_tot.subset('ae',0,cut[0]).get_energies(), bins=calVec, histtype='step', label='Spettro AE')
    plt.semilogy()
    plt.grid(axis='x')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    plt.xlim(lim[0],lim[1])
    plt.legend(loc='upper left')
    plt.semilogy()
    plt.show()
    return
            

if __name__ == '__main__':
    main()
    
    
"""
    #Print the fractions
    bounds=np.where((e_or>1850) & (e_or<2000))
      
    print('\n')
    print('NN vs Original')
    print('Double escape: '+str(evaluate_fraction(double_escape_NN,double_escape_or)[0])+' +- '+str(evaluate_fraction(double_escape_NN,double_escape_or)[1]))
    print('PhotoPeak 212Bi: '+str(evaluate_fraction(peakBi_NN,peakBi_or)[0])+' +- '+str(evaluate_fraction(peakBi_NN,peakBi_or)[1]))
    print('First escape: '+str(evaluate_fraction(first_escape_NN,first_escape_or)[0])+' +- '+str(evaluate_fraction(first_escape_NN,first_escape_or)[1]))
    print('PhotoPeak 208Tl: '+str(evaluate_fraction(peakTl_NN,peakTl_or)[0])+' +- '+str(evaluate_fraction(peakTl_NN,peakTl_or)[1]))
    
    
    print('\n')
    print('AE vs Original')
    print('Double escape: '+str(evaluate_fraction(double_escape_ae,double_escape_or)[0])+' +- '+str(evaluate_fraction(double_escape_ae,double_escape_or)[1]))
    print('PhotoPeak 212Bi: '+str(evaluate_fraction(peakBi_ae,peakBi_or)[0])+' +- '+str(evaluate_fraction(peakBi_ae,peakBi_or)[1]))
    print('First escape: '+str(evaluate_fraction(first_escape_ae,first_escape_or)[0])+' +- '+str(evaluate_fraction(first_escape_ae,first_escape_or)[1]))
    print('PhotoPeak 208Tl: '+str(evaluate_fraction(peakTl_ae,peakTl_or)[0])+' +- '+str(evaluate_fraction(peakTl_ae,peakTl_or)[1]))
    
    print('\n')
    print('NN vs AE')
    print('Double escape: '+str(evaluate_fraction(double_escape_NN,double_escape_ae)[0])+' +- '+str(evaluate_fraction(double_escape_NN,double_escape_ae)[1]))
    print('PhotoPeak 212Bi: '+str(evaluate_fraction(peakBi_NN,peakBi_ae)[0])+' +- '+str(evaluate_fraction(peakBi_NN,peakBi_ae)[1]))
    print('First escape: '+str(evaluate_fraction(first_escape_NN,first_escape_ae)[0])+' +- '+str(evaluate_fraction(first_escape_NN,first_escape_ae)[1]))
    print('PhotoPeak 208Tl: '+str(evaluate_fraction(peakTl_NN,peakTl_ae)[0])+' +- '+str(evaluate_fraction(peakTl_NN,peakTl_ae)[1]))
    
    print('\n')
    print('PhotoPeak 208Tl/Compton')
    bounds=np.where((e_or>1850) & (e_or<2000))
    print('Original: '+str(evaluate_fraction(peakTl_or,(np.sum(c_or[bounds]),0))[0])+' +- '+str(evaluate_fraction(peakTl_or,(np.sum(c_or[bounds]),0))[1]))
    bounds=np.where((e_ae>1850) & (e_ae<2000))
    print('AE: '+str(evaluate_fraction(peakTl_ae,(np.sum(c_ae[bounds]),0))[0])+' +- '+str(evaluate_fraction(peakTl_ae,(np.sum(c_ae[bounds]),0))[1]))
    bounds=np.where((e_NN>1850) & (e_NN<2000))
    print('NN: '+str(evaluate_fraction(peakTl_NN,(np.sum(c_NN[bounds]),0))[0])+' +- '+str(evaluate_fraction(peakTl_NN,(np.sum(c_NN[bounds]),0))[1]))
    
    print('\n')
    print('Compton NN/Compton Orig. : '+str(np.sum(c_NN[bounds])/np.sum(c_or[bounds])))
    print('Compton ae/Compton Orig. : '+str(np.sum(c_ae[bounds])/np.sum(c_or[bounds])))
    """
