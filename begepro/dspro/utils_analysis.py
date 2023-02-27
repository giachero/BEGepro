import numpy as np
from begepro.dspro import bege_event
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math

import IPython 

class analysis(object):
    def __init__(self,calVec):
        self.calVec=calVec
        return
            
    def histogram(self,coll,cutAI,comparison_energy,energy_range,name='AI'):
            
        c_AI, e_AI, p_AI =plt.hist(coll.subset('labels',0,cutAI).get_energies(), bins=self.calVec, histtype='step')
        c_counts=np.max(c_AI[np.where((e_AI>comparison_energy[0]) & (e_AI<comparison_energy[1]))[0]])
            
        ae=coll.get_avse()
        ae_max=max(ae)
        ae_min=min(ae) 
        cut=self.findCutAE(ae_min,ae_max,c_counts,coll,comparison_energy)
        print('cut AE: '+str(cut[0]))
        
        plt.figure()
        c_or, e_or, p_or = plt.hist(coll.get_energies(), bins=self.calVec, histtype='step', label='Spettro originario')
        c_AI, e_AI, p_AI = plt.hist(coll.subset('labels',0,cutAI).get_energies(), bins=self.calVec, histtype='step', label='Spettro '+name)
        c_ae, e_ae, p_ae = plt.hist(coll.subset('ae',0,cut[0]).get_energies(), bins=self.calVec, histtype='step', label='Spettro AE')
        plt.semilogy()
        plt.grid(axis='x')
        plt.xlabel('Energy [keV]')
        plt.ylabel('Counts')
        plt.xlim(energy_range[0],energy_range[1])
        plt.legend(loc='upper left')
        plt.semilogy()
        plt.show()
        return
        
    def findCutAE(self,ae_min,ae_max,c_counts,coll,comparison_energy,eps=0.001):
   
        while(True):
            cut=(ae_max+ae_min)/2
            coll_ae=coll.subset('ae',0,cut)
       
            c, e, p = plt.hist(coll_ae.get_energies(), bins=self.calVec, histtype='step')
            ae_counts=np.max(c[np.where((e>comparison_energy[0]) & (e<comparison_energy[1]))[0]])

            plt.close('all')
            
            try:
                val=abs(ae_counts/c_counts-1)
            except:
                print('frac: '+str(val))
                print('ae_counts: '+str(ae_counts))
                print('c_counts: '+str(c_counts))

            if(val<eps):
                break
            elif(ae_counts>c_counts):
                ae_max=cut
            else:
                ae_min=cut
                        
        return cut,ae_counts


    def AIvsAE(self,coll,peak,comparison_energy,compton,k):
        
        keys=list(peak.keys())
        points=np.zeros((len(keys),3,len(k)))
        cuts=np.zeros(len(k))
        
        for i in range(0,len(k)):
            cutAI=k[i]
                
            #CUT AI         
                
            #plt.figure()
            c_AI, e_AI, p_AI =plt.hist(coll.subset('labels',0,cutAI).get_energies(), bins=self.calVec, histtype='step')           
            c_counts=np.max(c_AI[np.where((e_AI>comparison_energy[0]) & (e_AI<comparison_energy[1]))[0]])

            #CUT AE
                
            ae=coll.get_avse()
            ae_max=max(ae)
            ae_min=min(ae) 
            cut=self.findCutAE(ae_min,ae_max,c_counts,coll,comparison_energy)
            cuts[i]=cut[0]
               
            #Spectrum
                
            plt.figure()
            c_or, e_or, p_or = plt.hist(coll.get_energies(), bins=self.calVec, histtype='step')
            c_AI, e_AI, p_AI = plt.hist(coll.subset('labels',0,cutAI).get_energies(), bins=self.calVec, histtype='step')
            c_ae, e_ae, p_ae = plt.hist(coll.subset('ae',0,cut[0]).get_energies(), bins=self.calVec, histtype='step')
                   
            plt.close('all')
                
            #ANALISIS
            res=self.analysis(peak,c_or, e_or,c_AI, e_AI,c_ae, e_ae)
            
            for j in range(0,len(keys)):
                pc=self.peak_compton(res[j][0],res[j][1],res[j][2],compton,c_or,e_or,c_AI,e_AI,c_ae,e_ae)
                points[j][0][i]=pc['original'][0]
                points[j][1][i]=pc['AI'][0]
                points[j][2][i]=pc['ae'][0]
            
            print('done '+str((i+1)/len(k)*100)+'%')
        
        markers=('o','x','s')
        colors=('blue','green','red')        
        plt.figure()
        
        for i in range(0,len(keys)):
            plt.scatter(k,points[i][0],marker=markers[0],c=colors[i])
            plt.scatter(k,points[i][1],marker=markers[1],c=colors[i])
            plt.scatter(k,points[i][2],marker=markers[2],c=colors[i])
            for n in range(0,len(points[i][1])):
                plt.text(k[n]+0.02,points[i][1][n],'{0:.5g}'.format(points[i][1][n]/points[i][2][n]*100))
           
        for i in range(0,len(cuts)):
                plt.text(k[i],-0.03,str(np.round(cuts[i],4)),rotation='vertical')
        
                    
        handles=list()
        red_patch=mpatches.Patch(color='red',label='208Tl')
        blue_patch=mpatches.Patch(color='blue',label='Double escape')
        green_patch=mpatches.Patch(color='green',label='212Bi')
        o_line=mlines.Line2D([],[],color='black',marker='o',markersize=10,ls="",label='Original')
        AI_line=mlines.Line2D([],[],color='black',marker='x',markersize=10,ls="",label='AI')
        AE_line=mlines.Line2D([],[],color='black',marker='s',markersize=10,ls="",label='ae')
        handles.append(red_patch)
        handles.append(blue_patch)
        handles.append(green_patch)
        handles.append(o_line)
        handles.append(AI_line)
        handles.append(AE_line)
        plt.legend(handles=handles)
        plt.xlabel('Cut AI')
        plt.ylabel('n_peak / Compton')
        plt.show()
    
        return
    
    def analysis(self,peak,c_or,e_or,c_AI,e_AI,c_ae,e_ae):

        keys=list(peak.keys())
        res=np.zeros((len(keys),3))
        
        for i in range(0,len(keys)):
            p=peak[keys[i]]
            res[i][0]=np.max(c_or[np.where((e_or>p[0]) & (e_or<p[1]))[0]])
            res[i][1]=np.max(c_AI[np.where((e_AI>p[0]) & (e_AI<p[1]))[0]])
            res[i][2]=np.max(c_ae[np.where((e_ae>p[0]) & (e_ae<p[1]))[0]])
        
        return res
        
    def peak_compton(self,peak_or,peak_AI,peak_ae,compton,c_or,e_or,c_AI,e_AI,c_ae,e_ae):

        #PhotoPeak / Compton
        bounds=np.where((e_or>compton[0]) & (e_or<compton[1]))
        original=self.evaluate_fraction((peak_or,0),(np.sum(c_or[bounds]),0))
        bounds=np.where((e_AI>compton[0]) & (e_AI<compton[1]))
        AI=self.evaluate_fraction((peak_AI,0),(np.sum(c_AI[bounds]),0))
        bounds=np.where((e_ae>compton[0]) & (e_ae<compton[1]))
        ae=self.evaluate_fraction((peak_ae,0),(np.sum(c_ae[bounds]),0))
        
        """
        original=evaluate_fraction((peak_or,0),(np.sum(c_or[bounds]),0))[0] , evaluate_fraction((peak_or,0),(np.sum(c_or[bounds]),0))[1]
        
        bounds=np.where((e_ae>1850) & (e_ae<2000))
        ae=evaluate_fraction((peak_ae,0),(np.sum(c_ae[bounds]),0))[0] , evaluate_fraction((peak_ae,0),(np.sum(c_ae[bounds]),0))[1]
        bounds=np.where((e_AI>1850) & (e_AI<2000))
        AI=evaluate_fraction((peak_AI,0),(np.sum(c_AI[bounds]),0))[0] , evaluate_fraction((peak_AI,0),(np.sum(c_AI[bounds]),0))[1]
        """
        return {'original'  : original,
                'AI'        : AI,
                'ae'        : ae}
        
    def evaluate_fraction(self,a,b):
        r=a[0]/b[0]
        err=math.sqrt((a[1]/b[0])**2+(a[0]*b[1]/(b[0])**2)**2)
        return (r,err)    
    
    
    
    
    """
    {'double_escape_or': double_escape_or,
            'peakBi_or': peakBi_or,
            'first_escape_or': first_escape_or,
            'peakTl_or'   : peakTl_or,
            'double_escape_AI': double_escape_AI,
            'peakBi_AI': peakBi_AI,
            'first_escape_AI': first_escape_AI,
            'peakTl_AI'   : peakTl_AI,
            'double_escape_ae': double_escape_ae,
            'peakBi_ae': peakBi_ae,
            'first_escape_ae': first_escape_ae,
            'peakTl_ae'   : peakTl_ae}    
    """    
             
        
        
        
        
