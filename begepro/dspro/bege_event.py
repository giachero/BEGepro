import numpy as np
                    
class BEGeEvent(object):
    def __init__(self, n_trace, dim_trace, trace=None, pheight=None, energy=None, amplitude=None, ae=None, index=None):

        self.__data={'trace'    : np.zeros([n_trace,dim_trace]).astype(np.int16)  if trace     is None else np.array(trace).astype(np.int16), 
                     'pheight'  : np.zeros([n_trace]).astype(np.int16)            if pheight   is None else np.array(pheight).astype(np.int16),
                     'energy'   : np.zeros([n_trace]).astype(np.float64)          if energy    is None else np.array(energy).astype(np.int16),
                     'amplitude': np.zeros([n_trace]).astype(np.int16)            if amplitude is None else np.array(amplitude).astype(np.int16),
                     'ae'       : np.zeros([n_trace]).astype(np.float64)          if ae        is None else np.array(ae).astype(np.int16),
                     'index'    : np.zeros([n_trace]).astype(np.int16)            if index     is None else np.array(index).astype(np.int16)}
                     
        self.__traces=0
        
        
        return

    def subset(self,index):    
        return BEGeEvent(0, 0,
                         self.get_traces()[index,:],
                         self.get_pulse_heights()[index],
                         self.get_energies()[index],
                         self.get_amplitudes()[index],
                         self.get_avse()[index],
                         self.get_indexes()[index])
    
    def update(self, key, value):
        if key in self.__data.keys():       
            self.__data[key]=value
            
        return
        
    def update_index(self):
        self.__data['index']=np.array(range(len(self.__data['trace'])))   
        return

    def __add_element(self, key, value):
        if key in self.__data.keys():
            self.__data[key][self.__traces]=value
            if key=='trace':      
                self.__traces+=1
            
        return

    def add_trace(self, trace):
        return self.__add_element('trace', np.array(trace).astype(np.int16))

    def add_pulse_height(self, pheight):
        return self.__add_element('pheight', np.int16(pheight))

    def add_energy(self, energy):
        return self.__add_element('energy', np.float64(energy))

    def add_amplitude(self, amplitude):
        return self.__add_element('amplitude', np.int16(amplitude))

    def add_avse(self, ae):
        return self.__add_element('ae', np.float64(ae))

    def get_data(self, key):
        return self.__data[key] if key in self.__data else None

    def get_traces(self):
        return self.get_data('trace')

    def get_pulse_heights(self):
        return self.get_data('pheight')

    def get_energies(self):
        return self.get_data('energy')
    
    def get_amplitudes(self):
        return self.get_data('amplitude')

    def get_avse(self):
        return self.get_data('ae')
        
    def get_indexes(self):
        return self.get_data('index')
        
    def set_trace(self,trace):
        self.__data['trace']=trace
        return

    def get_parameters(self):
        if all(k in self.__data.keys() for k in ['pheight', 'energy', 'amplitude', 'ae', 'index' ]):
            return np.transpose(np.array([self.__data['index'],
                                          self.__data['pheight'],
                                          self.__data['energy'],
                                          self.__data['amplitude'],
                                          self.__data['ae']]))
        else:
            return None        
                                
                        
                                       
                                    
        
    
