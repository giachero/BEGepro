import numpy as np
from scipy.signal import savgol_filter
import peakutils as pu
import re
from scipy.signal import find_peaks
from scipy.integrate import simpson
from begepro.dspro import bege_event

class rise_time(object):
    def __init__(self):
        return
            
    def compute_rt(self,trace,frequency,riseTimeLimits=(0.1,0.9)):
        i_m=np.where(trace==0)[0][0]
        i_min2=np.where(trace[i_m:]-riseTimeLimits[0]>=0)[0][0]+i_m
        i_max2=np.where(trace[i_m:]-riseTimeLimits[1] >=0)[0][0]+i_m
        
        i_min1=i_min2-1
        i_max1=i_max2-1
        
        self.t_min=np.interp(riseTimeLimits[0],[trace[i_min1],trace[i_min2]],[i_min1,i_min2])
        self.t_max=np.interp(riseTimeLimits[1],[trace[i_max1],trace[i_max2]],[i_max1,i_max2])

        res=(self.t_max-self.t_min)*frequency, tuple([i_min1,i_max2])
                    
        return res
        
class simm(object):
    def __init__(self):
        self.conf=np.load('/home/marco/work/Data/meanSSE.npy')
        o_rt=rise_time()
        rt,t=o_rt.compute_rt(self.conf,4e-9)
        
        o_der=second_derivative()
        der=o_der.compute_der(self.conf)
        der=o_der.compute_der(der)
        self.m_conf=np.where(der==np.min(der))[0][0]    
        return
        
    def compute_simm(self,trace):        
        trace=savgol_filter(trace,30,0)
        d=np.sum(abs(trace-self.conf))
        return d
  
    def compute_simm2(self,trace,der2):        
        trace=savgol_filter(trace,30,0)
        m_trace=np.where(der2==min(der2))[0][0]
        
        diff=abs(m_trace-self.m_conf)
        if(m_trace>self.m_conf):
            v=trace[diff:]
            simm=sum(abs(self.conf[:len(self.conf)-diff]-v))
        else:
            v=self.conf[diff:]
            simm=sum(abs(trace[:len(self.conf)-diff]-v))
 
        return simm
        
class second_derivative(object):
    def __init__(self):
        return
                
    def compute_n_zeros(self, f, t):
        l=t[1]-t[0]
        x=np.full(l,1)
        y=np.zeros(l)
        a=np.where(f[t[0] : t[1]]>=0,x,y)
        a=str(a)
        a=a.translate({ord(i): None for i in '[]. \n'})
        n=a.count('01')+a.count('10')
        
        indexes=[] 
        for match in re.finditer('01',a):
            indexes.append(match.start())
        for match in re.finditer('10',a):
            indexes.append(match.start())
            
        indexes=np.array(indexes)+t[0]
        
        cond=True
        while(cond):
            diff=abs(indexes[0:indexes.size-1]-indexes[1:])
            cut=np.where(diff<15)[0]
            if(cut.size>0):
                indexes=np.delete(indexes,cut[0])
                cond=True
            else:
                cond=False
                    
        return indexes
        
    def compute_der(self, f):
        smooth = savgol_filter(f, 14,0)
        der = savgol_filter(smooth, 2, 1, deriv=1)
        der = savgol_filter(der, 30, 0)
        return der, smooth
        
    def compute_area(self,f):
        
        return simpson(f/max(f))
        
    def compute_n_peaks(self,f,t):
        der2=abs(f)
        der2=der2/max(f)   
        indexes=find_peaks(f,distance=10,height=0.1)[0]
        return indexes.astype(np.int16)
             
        
        
        
        
