import numpy as np
from scipy.signal import savgol_filter
import peakutils as pu

class rise_time(object):
    def __init__(self):
        return
    
    def compute_rt(self,trace,frequency,riseTimeLimits=(0.1,0.9)):
        ylim=trace[-1]-trace[0]
        self.i_min=np.where(trace-trace[0]-riseTimeLimits[0]*ylim >=0)[0][0]
        self.i_max=np.where(trace-trace[0]-riseTimeLimits[1]*ylim >=0)[0][0]
        
        T=np.arange(0,len(trace))*frequency
        return T[self.i_max]-T[self.i_min]
        
class n_peaks(object):
    def __init__(self):
        return
    
    def compute_n_peaks(self,curr):       
        f=savgol_filter(curr,10,0)
        return pu.indexes(f,thres=0.3,min_dist=10)
