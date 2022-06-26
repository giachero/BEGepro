#!/usr/bin/env python

import numpy as np
import pylab as plt
import matplotlib.colors as clr
import math
from begepro.rw import CAENhandler_new as ca

from begepro.dspro import histfit as hf

import IPython

def main():

    for i in range(55,155,5):
        filename='/home/marco/work/tesi/data/228Th-grafico-tesi-im260421_1/AnalisiBig3/228Th-grafico-tesi-im260421_1__'+str(i)+'.npy'
        coll=ca.NPYreader(filename,False).get_event()
        labels=np.squeeze(np.load('/home/marco/work/tesi/data/NN/Th238/LastAnalysis/labels__'+str(i)+'.npy'))
        coll.set_labels(labels)
        coll=coll.subset('energy',1550)
        if(i==55):
            coll_tot=coll
        else:
            coll_tot=coll_tot+coll
            
        print('opened '+str(i))
    
    print('I am using :'+str(coll_tot.n_trace)+' events')
    
    plt.figure()
    plt.hist(coll_tot.get_labels(),color='b',alpha=1,bins=np.arange(0,1.1,0.005),density=True)
    plt.title('Distribution of the events')
    plt.xlabel('Neural Network label')
    plt.ylabel('Normalized counts')
    plt.xticks(np.arange(0,1.1,0.1))
    
    plt.text(0.03,-2,'MSE')
    plt.text(0.95,-2,'SSE')
    
    plt.show() 
    
    calVec = [-0.090383 + 0.20574*i for i in range(2**14+1)]
            
    #Spectrum
    #Here i find the optimal cuts for label and ae
    """
    plt.figure()
    c, e, p = plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='Spettro originario')
    plt.semilogy()
    plt.grid(axis='x')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    plt.xlim(1550,2650)
    plt.legend(loc='upper left')
    plt.semilogy()
    
    plt.show()
   
    peak = hf.HistogramFitter(c, e)
    xlim = 2090,2115

    shape = 'gaus'
    bkg = 'const'


    pars = {'ngaus': 10**4,
            'mu'   : 2104,
            'sigma': 2,
            'ntail': 10**3,
            'ttail': 2.5,
            'cstep': 50,
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    or_counts=peak.net_counts()[0]
    print(or_counts)
    
    peak.plot_fit()
    plt.show()
    
    
    
    labels=np.sort(coll_tot.get_labels())
    l=list(set(labels))    
    l=[x for x in l if x>0.8]
    l.sort()
       
    for i in l:
    
        coll_NN=coll_tot.subset('labels',0,1)
        c, e, p = plt.hist(coll_NN.get_energies(), bins=calVec, histtype='step', label='Spettro tagliato')
        peak = hf.HistogramFitter(c, e)
        xlim = 2090,2115

        shape = 'gaus'
        bkg = 'const'


        pars = {'ngaus': 10**4,
                'mu'   : 2104,
                'sigma': 2,                
                'p0'   : 10**1}
    
        peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
        peak.fit()
        NN_counts=peak.net_counts()[0]
        
    
        print('rapporto: '+str(NN_counts/or_counts)+' , cut: '+str(i))
                
        if(NN_counts/or_counts>=0.95):
            cut_NN=i
            break
            
            
    ae=np.sort(coll_tot.get_avse())
    l=list(set(ae))
    
    l=[x for x in l if ((x>0.017685935657475876) & (x<1.775e-2))]
    l.sort()
              
    for i in l:
    
        coll_ae=coll_tot.subset('ae',0,i)
        c, e, p = plt.hist(coll_ae.get_energies(), bins=calVec, histtype='step', label='Spettro tagliato')
        peak = hf.HistogramFitter(c, e)
        xlim = 2090,2115

        shape = 'gaus'
        bkg = 'const'


        pars = {'ngaus': 10**4,
                'mu'   : 2104,
                'sigma': 2,                
                'p0'   : 10**1}
    
        peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
        peak.fit()
        ae_counts=peak.net_counts()[0]
        
    
        print('rapporto: '+str(ae_counts/or_counts)+' , cut: '+str(i))
                
        if(ae_counts/or_counts>0.6909484505037544):
            cut_ae=i
            break
    """
    
    #Plot ae vs NN
    
    cut_NN=0.8000051975250244
    cut_ae=0.017686659029133446       
    """plt.figure()    
    plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='Spettro originario')
    c_NN, e_NN, p_NN =plt.hist(coll_tot.subset('labels',0,cut_NN).get_energies(), bins=calVec, histtype='step', label='Spettro tagliato NN, cut= '+str(cut_NN))
    c_ae, e_ae, p_ae =plt.hist(coll_tot.subset('ae',0,cut_ae).get_energies(), bins=calVec, histtype='step', label='Spettro tagliato ae cut= '+str(cut_ae),color='black',alpha=0.5)
    #plt.hist(coll_tot.subset('ae',0,0.0190).get_energies(), bins=calVec, histtype='step', label='Spettro tagliato da Alessandro cut= 0.0190',color='g',alpha=0.5)
    plt.semilogy()
    plt.xlim(1550,2650)
    plt.legend(loc='upper left')
    plt.title('Comparison between cuts')
    plt.show()
    
    #Fit to quantify the differences
    
    #Original

    plt.figure()    
    c_or, e_or, p_or =plt.hist(coll_tot.get_energies(), bins=calVec, histtype='step', label='Spettro Originale')
    plt.xlim(1550,2650)
    #plt.semilogy()
    
    #Double escape
    peak = hf.HistogramFitter(c_or, e_or)
    xlim = 1590,1600

    shape = 'gaus'
    bkg = 'const'

    pars = {'ngaus': 1750,
            'mu'   : 1593,
            'sigma': 2,           
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    double_escape_or=peak.net_counts()
    print('done de') 
    
    #Peak 212Bi
    peak = hf.HistogramFitter(c_or, e_or)
    xlim = 1615,1630

    shape = 'gaus'
    bkg = 'const'

    pars = {'ngaus': 1250,
            'mu'   : 1621,
            'sigma': 2,            
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    peakBi_or=peak.net_counts()
    print('done Bi') 
    
    #Peak first escape
    peak = hf.HistogramFitter(c_or, e_or)
    xlim = 2090,2120

    shape = 'gaus'
    bkg = 'const'

    pars = {'ngaus': 1300,
            'mu'   : 2104,
            'sigma': 2,
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    first_escape_or=peak.net_counts()
    print('done fe')    
        
    #Peak 208Tl
    peak = hf.HistogramFitter(c_or, e_or)
    xlim = 2580,2640

    shape = 'hyper'
    bkg = 'const'

    pars = {'ngaus': 10**4,
            'mu'   : 2615,
            'sigma': 2,
            'ntail': 10**3,
            'ttail': 2.5,
            'cstep': 50,
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    peakTl_or=peak.net_counts()
    print('done Tl')
        
    plt.show()
    
    #NN
    
    plt.figure()    
    c_NN, e_NN, p_NN =plt.hist(coll_tot.subset('labels',0,cut_NN).get_energies(), bins=calVec, histtype='step', label='Spettro tagliato NN, cut= '+str(cut_NN))
    plt.xlim(1550,2650)
    #plt.semilogy()
    
    #Double escape
    peak = hf.HistogramFitter(c_NN, e_NN)
    xlim = 1590,1600

    shape = 'gaus'
    bkg = 'const'

    pars = {'ngaus': 300,
            'mu'   : 1593,
            'sigma': 2,
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    double_escape_NN=peak.net_counts() 
    
    print('NN: '+str(peak.get_parameters()))
    
    #Peak 212Bi
    peak = hf.HistogramFitter(c_NN, e_NN)
    xlim = 1615,1630

    shape = 'gaus'
    bkg = 'const'

    pars = {'ngaus': 800,
            'mu'   : 1621,
            'sigma': 2,            
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    peakBi_NN=peak.net_counts()
    
    #Peak first escape
    peak = hf.HistogramFitter(c_NN, e_NN)
    xlim = 2090,2120

    shape = 'gaus'
    bkg = 'const'

    pars = {'ngaus': 800,
            'mu'   : 2104,
            'sigma': 2,            
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    first_escape_NN=peak.net_counts()   
        
    #Peak 208Tl
    peak = hf.HistogramFitter(c_NN, e_NN)
    xlim = 2580,2640

    shape = 'hyper'
    bkg = 'const'

    pars = {'ngaus': 6500,
            'mu'   : 2615,
            'sigma': 2,
            'ntail': 10**3,
            'ttail': 2.5,
            'cstep': 50,            
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    peakTl_NN=peak.net_counts()
        
    plt.show()
        
    #AE
    
    plt.figure()    
    c_ae, e_ae, p_ae =plt.hist(coll_tot.subset('ae',0,cut_ae).get_energies(), bins=calVec, histtype='step', label='Spettro tagliato AE, cut= '+str(cut_ae))
    plt.xlim(1550,2650)
    #plt.semilogy()
    
    #Double escape
    peak = hf.HistogramFitter(c_ae, e_ae)
    xlim = 1590,1600

    shape = 'gaus'
    bkg = 'const'

    pars = {'ngaus': 70,
            'mu'   : 1593,
            'sigma': 2,            
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    double_escape_ae=peak.net_counts() 
    
    #Peak 212Bi
    peak = hf.HistogramFitter(c_ae, e_ae)
    xlim = 1615,1630

    shape = 'gaus'
    bkg = 'const'

    pars = {'ngaus': 700,
            'mu'   : 1621,
            'sigma': 2,
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    peakBi_ae=peak.net_counts()
    
    #Peak first escape
    peak = hf.HistogramFitter(c_ae,e_ae)
    xlim = 2090,2120

    shape = 'gaus'
    bkg = 'const'

    pars = {'ngaus': 800,
            'mu'   : 2104,
            'sigma': 2,    
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    first_escape_ae=peak.net_counts()   
        
    #Peak 208Tl
    peak = hf.HistogramFitter(c_ae, e_ae)
    xlim = 2580,2640

    shape = 'hyper'
    bkg = 'const'

    pars = {'ngaus': 6000,
            'mu'   : 2615,
            'sigma': 2,
            'ntail': 10**3,
            'ttail': 2.5,
            'cstep': 50,
            'p0'   : 10**1}
    
    peak.set_model((shape,bkg), xlim=xlim, initpars=pars)
    peak.fit()
    peak.plot_fit()
    peakTl_ae=peak.net_counts()
        
    plt.show()
    
    #Print the fractions
    
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
  
    """#Other plots  
    
    #coll_SSE=coll_tot.subset('labels',cut_NN)
    #coll_MSE=coll_tot.subset('labels',0,cut_NN)      
    
    #%matplotlib widget
    fig,axs=plt.subplots(2,figsize=(9,20))
    axs[0].hist2d(coll_tot.subset('energy',1550,2650).subset('labels',0,0.02).get_energies(), coll_tot.subset('energy',1550,2650).subset('labels',0,0.02).get_labels(),
            bins=[calVec,250], range=[(calVec[0],calVec[-1]),(0,0.02)], 
            cmin=1, cmap=plt.cm.turbo, norm=clr.LogNorm())
    axs[0].set_xlim([1550,2650])
    axs[1].hist2d(coll_tot.subset('energy',1550,2650).subset('labels',0.8,1).get_energies(), coll_tot.subset('energy',1550,2650).subset('labels',0.8,1).get_labels(),
            bins=[calVec,250], range=[(calVec[0],calVec[-1]),(0.8,1)], 
            cmin=1, cmap=plt.cm.turbo, norm=clr.LogNorm())
    axs[1].set_title('Current')
    axs[1].set_xlim([1550,2650])
    
    #plt.hist2d(coll_tot.get_energies(), coll_tot.get_labels(),
     #       bins=[calVec,250], range=[(calVec[0],calVec[-1]),(0,1)], 
      #      cmin=1, cmap=plt.cm.turbo, norm=clr.LogNorm())
    #plt.colorbar()
    plt.show()
    
    """      
    #%matplotlib widget
    plt.figure()
    plt.hist2d(coll_SSE.get_energies(), coll_SSE.get_avse(),
            bins=[calVec,250], range=[(calVec[0],calVec[-1]),(0.010,0.025)], 
            cmin=1, cmap=plt.cm.turbo, norm=clr.LogNorm())
    plt.colorbar()
    plt.xlabel('Energy [keV]')
    plt.ylabel('A/E') 
    plt.title('SSE')
    plt.xlim(1550,2650)
    
    plt.show()
    
    #Spectrum of SSE
    plt.figure()
    plt.hist(coll_SSE.get_energies(), bins=calVec, histtype='step', alpha=0.7)
    plt.semilogy()
    plt.grid(axis='x')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    plt.xlim(1550,2650)
    
    plt.show() 
    
    plt.figure()
    plt.hist2d(coll_MSE.get_energies(), coll_MSE.get_avse(),
            bins=[calVec,250], range=[(calVec[0],calVec[-1]),(0.010,0.025)], 
            cmin=1, cmap=plt.cm.turbo, norm=clr.LogNorm())
    plt.colorbar()
    plt.xlabel('Energy [keV]')
    plt.ylabel('A/E') 
    plt.title('MSE')
    plt.xlim(1550,2650)
    
    plt.show() 
    
    #Spectrum of MSE
    plt.figure()
    plt.hist(coll_MSE.get_energies(), bins=calVec, histtype='step', alpha=0.7)
    plt.semilogy()
    plt.grid(axis='x')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    plt.xlim(1550,2650)"""
     
    return
    
def evaluate_fraction(a,b):
    r=a[0]/b[0]
    err=math.sqrt((a[1]/b[0])**2+(a[0]*b[1]/(b[0])**2)**2)
    return (r,err)
    
    
if __name__ == '__main__':
    main()
