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
        
        
    def compute_rt2(self,trace,frequency,riseTimeLimits=(0.1,0.9)):
        #trace=savgol_filter(trace,20,0)
        M=max(trace)
        m=min(trace)
        i_m=np.where(trace==m)[0][0]
        ylim=M-m
        i_min2=np.where(trace[i_m:]-m-riseTimeLimits[0]*ylim >=0)[0][0]+i_m
        self.prova=i_min2
        i_min1=i_min2-1
        i_max2=np.where(trace-m-riseTimeLimits[1]*ylim >=0)[0][0]
        i_max1=i_max2-1
        
        self.t_min=np.interp(m+riseTimeLimits[0]*ylim,[trace[i_min1],trace[i_min2]],[i_min1,i_min2])
        self.t_max=np.interp(m+riseTimeLimits[1]*ylim,[trace[i_max1],trace[i_max2]],[i_max1,i_max2])

        res=(self.t_max-self.t_min)*frequency,tuple([int(np.ceil(self.t_min)),int(np.ceil(self.t_max))])
        """if(self.t_max)<0:
            import pylab as plt
            plt.figure()
            plt.plot(trace)
            plt.scatter(i_min2,trace[i_min2])
            print(np.where(trace-m-riseTimeLimits[0]*ylim >=0))
            plt.show()"""
            
        return res
        
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
        
    def compute_n_zeros(self,trace,t):
        #f=savgol_filter(trace,20,0)
        #f=savgol_filter(f,10,2,deriv=2) #maggiore ordine
        f=savgol_filter(trace,20,0)
        f=savgol_filter(f,10,1,deriv=1)
        f=savgol_filter(f,20,0)
        
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
            
        indexes=indexes+t[0]

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
                    
        return indexes
        
        
    def compute_n_zeros2(self,curr,t,c=None,f=None):

        if((c is None) | (f is None)):
            c,f=self.compute_der(curr)
        
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
        
    def compute_der(self,curr):
        c=savgol_filter(curr,14,0)
        f=savgol_filter(c,2,1,deriv=1)
        return c,f
        
    def compute_n_peaks(self,second_der,t):   
        indexes=find_peaks(second_der[t[0] : t[1]],distance=10)[0]+t[0]
        return indexes.astype(np.int16)

class simm(object):
    def __init__(self):
        return
    
    def compute_simm(self,trace):
    
        trace=savgol_filter(trace,30,0)
        trace=trace-min(trace)
        h=int(trace.size/2)
        t1=trace[0:h]
        t1=t1[::-1]
        t1=t1*(-1)
        t1=t1-min(t1)
        t2=trace[h:]
        t2=t2-np.min(t2)

        d=np.sum(abs((t1) - (t2)))
        return d
        
    def compute_simm2(self,curr):    
        curr=savgol_filter(curr,30,0)

        ind=np.where(curr==max(curr))[0][0]
        s=int((min(curr.size-ind,ind-1)/10))

        t1=curr[ind-s:ind]
        t1=t1-min(t1)
        #t1=t1/max(t1)
        t2=curr[ind:ind+s]
        t2=t2-min(t2)
        t2=t2[::-1]
        #t2=t2/max(t2)

        d=np.sum(abs(t1-t2))
        return d
        
    def compute_simm3(self,trace):
    
        conf=np.load('/home/marco/work/tesi/data/TestSimm/trace.npy')
    
        trace=savgol_filter(trace,30,0)
        trace=trace-min(trace)
        trace=trace/max(trace)

        d=np.sum(abs(trace-conf))
        return d
        
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
        
        
        
        
        
        
