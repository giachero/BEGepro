import numpy as np

class BEGeEvent(object):
    def __init__(self, trace=None, pheight=None, energy=None, amplitude=None, ae=None):  #underscore dopo nome?

        self.__data={'trace'    : list() if trace     is None else trace,
                     'pheight'  : list() if pheight   is None else pheight,
                     'energy'   : list() if energy    is None else energy,
                     'amplitude': list() if amplitude is None else energy
                     'ae       ': list() if ae        is None else ae}
        
        return


    def update(self, key, value):
        if key in self._data[key]:
            self.__data[key]=value
            
        return

    def __add_element(self, key, value):
        if key in self._data[key]:
            self.__data[key].append(value)
        return

    def add_trace(self, trace):
        return self.__add_element(self, 'trace', trace)

    def add_pulse_height(self, pheight):
        return self.__add_element(self, 'pheight', pheight)

    def add_energy(self, energy):
        return self.__add_element(self, 'energy', energy)

    def add_amplitude(self, amplitude):
        return self.__add_element(self, 'amplitude', amplitude)

    def add_avse(self, ae):
        return self.__add_element(self, 'ae', ae)

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


    def get_parameters(self):
        if all(k in ['pheight', 'energy', 'amplitude', 'ae' ] for k in self.__data.keys()):
            return np.transpose(np.array([self.__data['pheight'],
                                          self.__data['energy'],
                                          self.__data['amplitude'],
                                          self.__data['ae']]))
        else:
            return None
                                
                        
                                       
                                    
        
    
