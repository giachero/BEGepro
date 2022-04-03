import numpy as np
from scipy.signal import savgol_filter
import peakutils as pu
from scipy.signal import find_peaks
from begepro.dspro import bege_event

class rise_time(object):
    def __init__(self):
        return
    
    def compute_rt(self,trace,frequency,riseTimeLimits=(0.1,0.9)):
        ylim=trace[-1]-trace[0]
        self.i_min=np.where(trace-trace[0]-riseTimeLimits[0]*ylim >=0)[0][0]
        self.i_max=np.where(trace-trace[0]-riseTimeLimits[1]*ylim >=0)[0][0]
        
        T=np.arange(0,len(trace))*frequency
        return T[self.i_max]-T[self.i_min],tuple([self.i_min,self.i_max])
        
class n_peaks(object):
    def __init__(self):
        return
    
    def compute_n_peaks(self,curr):       
        f=savgol_filter(curr,10,0)
        return pu.indexes(f,thres=0.3,min_dist=10)
        
    def compute_n_peaks2(self,curr,t):       
        indexes=find_peaks(curr,prominence=1,height=0.25*max(curr),distance=30)[0]
        return indexes.astype(np.int16) 
        
        #prominence ad occhio guardando low energy
        #height 50% del picco max
        # distanza 30 impostata guardando i=120
        
        
class plateau(object):
    def compute_plateau(self,coll1,l):
        p=0.001
        arr=list()
        for j in range (0,coll1.n_trace):
            curr=coll1.get_curr()[j]
            trace=coll1.get_traces()[j]
            rt,t=rise_time().compute_rt(trace,4e-9)
            val=max(curr)*p
            for i in range (t[0],t[1]):
                c=0
                for k in range(1,l+1):
                    if (abs(curr[i+k]-curr[i])<val):
                        c+=1
                    else:
                        break;
                    if (c==l):
                        #print('plateau at '+str(i)+' file '+str(j))
                        arr=[*arr,[j,i,t]]
        i=0
        while(i<len(arr)-1):
            if((arr[i][0]==arr[i+1][0]) & (abs(arr[i][1]-arr[i+1][1])==1)):
                arr.pop(i)
                #print('r')
            else:
                i+=1
        return arr
        
        
    def compute_plateau2(self,curr,l):
        p=0.001
        arr=list()    
        val=max(curr)*p
        cond=False
        
        for i in range (t[0],t[1]):
            c=0
            if(cond):
                cond=False
                continue;
            for k in range(1,l+1):
                if (abs(curr[i+k]-curr[i])<val):
                    c+=1
                else:
                    break;
                if (c==l):
                    arr=[*arr,[i]]
                    cond=True
        i=0
        while(i<len(arr)-1):
            if((arr[i][0]==arr[i+1][0]) & (abs(arr[i][0]-arr[i+1][0])==1)):
                arr.pop(i)
            else:
                i+=1
        return arr
        
        
    def compute_plateau3(self,curr,t,index):
        p=0.004
        l=7

        res=list()
        diff=np.diff(curr)
        val=max(curr)*p
        c=0
        for i in range(t[0],t[1]+1): 
            diff=abs(curr-curr[i])
            cond=diff<val
            o=np.diff(np.where(np.concatenate(([cond[0]],cond[:-1] != cond[1:], [True])))[0])[::2]
            for k in o:
                if (k>l): 
                    #print(str(i)+' of lenght '+str(k))
                    res=res+list([(i,k,index)])
        if(len(res)>0):
            if(all(np.equal(np.diff(np.array(res)[:,0]),np.full(len(res)-1,1)))):
                res=np.array(list([(res[0][0],res[-1][0]-res[0][0])]))

        return res
        
        
        
        
        
        
