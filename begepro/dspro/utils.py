import numpy as np
from scipy.signal import savgol_filter
import peakutils as pu
import re
from scipy.signal import find_peaks
from begepro.dspro import bege_event

class rise_time(object):
    def __init__(self):
        return
    
    def compute_rt(self,trace,frequency,riseTimeLimits=(0.1,0.9)):
        M=max(trace)
        m=min(trace)
        ylim=M-m
        self.i_min=np.where(trace-m-riseTimeLimits[0]*ylim >=0)[0][0]
        self.i_max=np.where(trace-m-riseTimeLimits[1]*ylim >=0)[0][0]
        
        T=np.arange(0,len(trace))*frequency
        return T[self.i_max]-T[self.i_min],tuple([self.i_min,self.i_max])
        
class n_peaks(object):
    def __init__(self):
        return
        
    def compute_n_peaks(self,curr,energy,E,maxlim):
    
        if(energy>E):
            p=0.5
            h=0.1    
        elif(max(curr)>=maxlim):
            p=0.5
            h=0.25
        else:
            p=0.9
            h=0.25
                       
        indexes=find_peaks(curr,prominence=p,height=h*max(curr),distance=10)[0]
        return indexes.astype(np.int16)
        
        #prominence 0.5 per E>750 KeV e 1.6 per <
        #height 0.10 per E>750 KeV e 0.25 per <
        # distanza 10 impostata guardando i=222
        
class second_derivative(object):
    def __init__(self):
        return
        
    def compute_n_zeros(self,t):
        l=rtobj.i_max-rtobj.i_min
        x=np.full(l,1)
        y=np.zeros(l)
        a=np.where(f[rtobj.i_min : rtobj.i_max]>=0,x,y)
        a=str(a)
        a=a.translate({ord(i): None for i in '[]. \n'})
        n=a.count('01')+a.count('10')

        indexes=[]
        for match in re.finditer('01',a):
            indexes.append(match.start())
        for match in re.finditer('10',a):
            indexes.append(match.start())
            
        indexes=indexes+rtobj.i_min

        cond=True
        while(cond & len(indexes)>1):
            #print('c')
            for i in range(0,len(indexes)-1):
                cond=False
                if indexes[i]>indexes[i+1]-5:
                    print(i)
                    indexes.pop(i)
                    cond=True
                    break;
                    
        print(len(indexes))
    
plt.scatter(indexes,f[indexes],marker='o',color='r')

        
        
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
        
        
        
        
        
        
