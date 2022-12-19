# Class:        BEGeEvent 
# Description:  useful class to store the pre-amplified pulses' parameters 
# Attributes:   >> traces:      number of the events that is actually containing
#               >> n_trace:     max number of the events that can contain
#               >> dim_trace:   number of points describing each pulse shape     
#               >> data:        dictionary containing all the informations about the pulses
#                               >>trace:        pre-amplified pulse of the event
#                               >>curr:         current signal obtained via differentiation
#                               >>pheight:      ???
#                               >>energy:       energy of the event
#                               >>amplitude:    amplitude of the current
#                               >>ae:           amplitude over energy
#                               >>risetime:     risetime of the trace
#                               >>zeros_2der:   number of zeros of the differentiated current
#                               >>n_peaks_2der: number of peaks of the differentiated current
#                               >>simm:         value of the simmetry parameter
#                               >>labels:       label of the neural network
#                               >>index:        enumeration of the event


#!!! salvare solo trace con autoencoder curr non serve. Risparmiamo ram  

import numpy as np
import math as math
                    
class BEGeEvent(object):
    def __init__(self, n_trace, dim_trace, trace=None,curr=None, pheight=None, energy=None, amplitude=None, ae=None, risetime=None,n_peaks=None,zeros_2der=None,n_peaks_2der=None,simm=None,labels=None,index=None):

        self.__data={'trace'       : np.zeros([n_trace,dim_trace]).astype(np.int16)   if trace        is None else np.array(trace).astype(np.int16),
                     'curr'        : np.zeros([n_trace,dim_trace]).astype(np.float64) if curr         is None else np.array(curr).astype(np.float64), 
                     'pheight'     : np.zeros([n_trace]).astype(np.int16)             if pheight      is None else np.array(pheight).astype(np.int16),
                     'energy'      : np.zeros([n_trace]).astype(np.float64)           if energy       is None else np.array(energy).astype(np.float64),
                     'amplitude'   : np.zeros([n_trace]).astype(np.int16)             if amplitude    is None else np.array(amplitude).astype(np.int16),
                     'ae'          : np.zeros([n_trace]).astype(np.float64)           if ae           is None else np.array(ae).astype(np.float64),
                     'risetime'    : np.zeros([n_trace]).astype(np.float64)           if risetime     is None else np.array(risetime).astype(np.float64),
                     'n_peaks'     : np.zeros([n_trace]).astype(np.int16)             if n_peaks      is None else np.array(n_peaks).astype(np.int16),
                     'zeros_2der'  : np.zeros([n_trace]).astype(np.int16)             if zeros_2der   is None else np.array(zeros_2der).astype(np.int16),
                     'n_peaks_2der': np.zeros([n_trace]).astype(np.int16)             if n_peaks_2der is None else np.array(n_peaks_2der).astype(np.int16),
                     'simm'        : np.zeros([n_trace]).astype(np.float64)           if simm         is None else np.array(simm).astype(np.float64),
                     'labels'      : np.zeros(n_trace).astype(np.float64)             if labels       is None else np.array(labels).astype(np.double),
                     'index'       : np.zeros([n_trace]).astype(np.int16)             if index        is None else np.array(index).astype(np.int32)}
                     
        self.__traces=0
        self.n_trace=n_trace
        self.dim_trace=dim_trace
        
        
        return

# Method:       subset
# Description:  performs cuts on a given parameter
# Returns:      BEGeEvent with values of the parameter "key" included between "cutmin" and "cutmax"
#               or
#               BEGeEvent with the events specified by the array "index"

    def subset(self,key,cutmin=None,cutmax=None,index=None):
        if((index is None) & (cutmin==None)):
            return
        if ((key in self.__data.keys())&((key!='trace') & (key!='curr'))): 
            if index is None:
                if cutmax==None: cutmax=float(math.inf)  
                index=np.where((self.get_data(key) >= cutmin) & (self.get_data(key) <= cutmax))[0]          
            return BEGeEvent(np.array(index).size,self.dim_trace,
                             self.get_traces()[index,:] if self.get_traces().size != 0 else None,
                             self.get_curr()[index,:] if self.get_curr().size != 0 else None,
                             self.get_pulse_heights()[index],
                             self.get_energies()[index],
                             self.get_amplitudes()[index],
                             self.get_avse()[index],
                             self.get_risetime()[index],
                             self.get_n_peaks()[index],
                             self.get_zeros_2der()[index],
                             self.get_n_peaks_2der()[index],
                             self.get_simm()[index],
                             self.get_labels()[index],
                             self.get_indexes()[index])
    
# Method:       update
# Description:  updates a specified element of the dictonary "data" specified by "key"
# Returns:      none

    def update(self, key, value):
        if key in self.__data.keys():       
            self.__data[key]=value
            
        return
        
# Method:       update_index
# Description:  updates the parameter "index" inside "data"
# Returns:      none
    
    def update_index(self):
        self.__data['index']=np.array(range(len(self.__data['n_peaks'])))   
        return

    def __add_element(self, key, value):
        if key in self.__data.keys():
            self.__data[key][self.__traces]=value
            if key=='n_peaks':      
                self.__traces+=1
            
        return
        
    def remove_zeros(self):
        v=np.where(self.get_avse()==0)[0]
        v=np.delete(np.arange(0,self.n_trace),v)
        return self.subset('ae',index=v)

    def add_trace(self, trace):
        return self.__add_element('trace', np.array(trace).astype(np.int16))
        
    def add_curr(self, curr):
        return self.__add_element('curr', np.array(curr).astype(np.float64))

    def add_pulse_height(self, pheight):
        return self.__add_element('pheight', np.int16(pheight))

    def add_energy(self, energy):
        return self.__add_element('energy', np.float64(energy))

    def add_amplitude(self, amplitude):
        return self.__add_element('amplitude', np.int16(amplitude))

    def add_avse(self, ae):
        return self.__add_element('ae', np.float64(ae))
        
    def add_risetime(self, risetime):
        return self.__add_element('risetime', np.float64(risetime))
        
    def add_n_peaks(self, n_peaks):
        return self.__add_element('n_peaks', np.array(n_peaks).astype(np.int16))
        
    def add_zeros_2der(self, zeros_2der):
        return self.__add_element('zeros_2der', np.array(zeros_2der).astype(np.int16))
        
    def add_n_peaks_2der(self, n_peaks_2der):
        return self.__add_element('n_peaks_2der', np.array(n_peaks_2der).astype(np.int16))
        
    def add_simm(self, simm):
        return self.__add_element('simm', np.float64(simm))
       
    def get_data(self, key):
        return self.__data[key] if key in self.__data else None
        
    def get_dict(self):
        return self.__data

    def get_traces(self):
        return self.get_data('trace')
        
    def get_curr(self):
        return self.get_data('curr')

    def get_pulse_heights(self):
        return self.get_data('pheight')

    def get_energies(self):
        return self.get_data('energy')
    
    def get_amplitudes(self):
        return self.get_data('amplitude')

    def get_avse(self):
        return self.get_data('ae')
        
    def get_risetime(self):
        return self.get_data('risetime')
        
    def get_n_peaks(self):
        return self.get_data('n_peaks')
        
    def get_zeros_2der(self):
        return self.get_data('zeros_2der')
        
    def get_n_peaks_2der(self):
        return self.get_data('n_peaks_2der')
        
    def get_simm(self):
        return self.get_data('simm')
        
    def get_labels(self):
        return self.get_data('labels')
        
    def get_indexes(self):
        return self.get_data('index')
        
    def get_n_elements(self):
        return self.__traces
        
    def set_trace(self,trace):
        self.__data['trace']=trace
        return
        
    def set_curr(self,curr):
        self.__data['curr']=curr
        return
        
    def set_labels(self,labels):
        self.__data['labels']=labels
        return
        
    def set_simm(self,simm):
        self.__data['simm']=simm
        return

    def get_parameters(self):
        if all(k in self.__data.keys() for k in ['pheight', 'energy', 'amplitude', 'ae', 'risetime','n_peaks','zeros_2der','n_peaks_2der','simm','index' ]):
            return np.transpose(np.matrix([self.__data['index'],
                                          self.__data['pheight'],
                                          self.__data['energy'],
                                          self.__data['amplitude'],
                                          self.__data['ae'],
                                          self.__data['risetime'],
                                          self.__data['n_peaks'],
                                          self.__data['zeros_2der'],
                                          self.__data['n_peaks_2der'],
                                          self.__data['simm']
                                          ]))
        else:
            return None
            
    def __add__(self,be):  
        return BEGeEvent(self.n_trace+be.n_trace,
                         self.dim_trace,
                         np.array(np.concatenate((self.get_traces(),be.get_traces()))),
                         np.array(np.concatenate((self.get_curr(),be.get_curr()))),
                         np.hstack((self.get_pulse_heights(),be.get_pulse_heights())),
                         np.hstack((self.get_energies(),be.get_energies())),
                         np.hstack((self.get_amplitudes(),be.get_amplitudes())),
                         np.hstack((self.get_avse(),be.get_avse())),
                         np.hstack((self.get_risetime(),be.get_risetime())),
                         np.hstack((self.get_n_peaks(),be.get_n_peaks())),
                         np.hstack((self.get_zeros_2der(),be.get_zeros_2der())),
                         np.hstack((self.get_n_peaks_2der(),be.get_n_peaks_2der())),
                         np.hstack((self.get_simm(),be.get_simm())),
                         np.hstack((self.get_labels(),be.get_labels())),
                         np.hstack((self.get_indexes(),be.get_indexes()))
                         )
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
                    
        
