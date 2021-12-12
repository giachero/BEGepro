import numpy as np
                    
class BEGeEvent(object):
    def __init__(self, trace=None, pheight=None, energy=None, amplitude=None, ae=None, index=None):

        self.__data={'trace'    : list() if trace     is None else trace,
                     'pheight'  : list() if pheight   is None else pheight,
                     'energy'   : list() if energy    is None else energy,
                     'amplitude': list() if amplitude is None else amplitude,
                     'ae'       : list() if ae        is None else ae,
                     'index'    : list() if index     is None else index}
        
        return

    def subset(self,index):
        trace=list()
        pheight=list()
        energy=list()
        amplitude=list()
        ae=list()
              
        index.sort()
            
        for i in index:
            try:
                n=self.__data["index"].index(i)
                
                trace.append(self.__data["trace"][n])
                pheight.append(self.__data["pheight"][n])
                energy.append(self.__data["energy"][n])
                amplitude.append(self.__data["amplitude"][n])
                ae.append(self.__data["ae"][n])
            except:
                print("index "+str(i)+" is not present")    

        return BEGeEvent(trace,pheight,energy,amplitude,ae,index)
    
    def update(self, key, value):
        if key in self.__data.keys():
            self.__data[key]=value
            
        return

    def __add_element(self, key, value):
        if key in self.__data.keys():
            self.__data[key].append(value)
            self.__data['index']=range(len(self.__data[key]))
        return

    def add_trace(self, trace):
        return self.__add_element('trace', trace)

    def add_pulse_height(self, pheight):
        return self.__add_element('pheight', pheight)

    def add_energy(self, energy):
        return self.__add_element('energy', energy)

    def add_amplitude(self, amplitude):
        return self.__add_element('amplitude', amplitude)

    def add_avse(self, ae):
        return self.__add_element('ae', ae)

    def get_data(self, key):
        return np.array(self.__data[key]) if key in self.__data else None

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


    def get_parameters(self):
        if all(k in self.__data.keys() for k in ['pheight', 'energy', 'amplitude', 'ae', 'index' ]):
            return np.transpose(np.array([self.__data['index'],
                                          self.__data['pheight'],
                                          self.__data['energy'],
                                          self.__data['amplitude'],
                                          self.__data['ae']]))
        else:
            return None
                                
                        
                                       
                                    
        
    
