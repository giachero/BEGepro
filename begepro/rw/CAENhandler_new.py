import os, errno 
import struct
import pylab as plt
import numpy as np

from sys import getsizeof

from lxml import etree as ET

from begepro.dspro import bege_event as be

class XMLreader(object):
    def __init__(self, filename):
        if not os.path.isfile(filename):
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

        #self.__context = ET.iterparse(filename,tag='trace')
        
        self.__context = ET.iterparse(filename,
                                      #events=('start','end'),
                                      tag=('event'))
        return


    def get(self):

        try:
            elem = next(self.__context)[1]
        except Exception as ex:
            if type(ex).__name__ in ['StopIteration', 'XMLSyntaxError']:
                return None
            else: 
                #print (type(ex).__name__, ex.args)
                raise ex
            
        ret=dict(elem.attrib)
        ret.setdefault('channels', {})
        
        for e in elem:
            if e.tag == 'trace':
                ret['channels'].update({int(e.attrib['channel']): list(map(float, e.text.rstrip('\n ').split())) })
        
        return ret

class NPYreader(object):
    def __init__(self, dir,filename,index,include_trace=False,include_curr=False):
        self.__dir=dir
        self.__filename = filename
        self.__index = str(index)
        path = dir + filename + '_' + self.__index + '.npy'
        matrix=np.load(path,allow_pickle=True)
        self.__event=be.BEGeEvent(len(matrix),
                                  len(matrix[0]), 
                                  trace=np.array([]),
                                  curr=np.array([]), 
                                  pheight=matrix[:,1], 
                                  energy=matrix[:,2], 
                                  amplitude=matrix[:,3], 
                                  ae=matrix[:,4], 
                                  risetime=matrix[:,5],           
                                  n_peaks=matrix[:,6],
                                  zeros_2der=matrix[:,7],
                                  n_peaks_2der=matrix[:,8],
                                  simm=matrix[:,9],
                                  index=matrix[:,0])
        if include_trace: self.set_trace()
        if include_curr:  self.set_curr()
        return
        
    def set_curr(self):
        curr_filename = self.__dir + self.__filename + "curr__" + self.__index + '.npy'
        curr=np.load(curr_filename,allow_pickle=True)
        self.__event.set_curr(curr)
        
    def set_trace(self):
        trace_filename=self.__dir + self.__filename + "trace__" + self.__index + '.npy'
        trace=np.load(trace_filename,allow_pickle=True)
        self.__event.set_trace(trace)
        
    def get_event(self):
        return self.__event
    
'''
enum style class
'''
class Debug(object):
    NO_DEBUG  = 0
    PLOT_DATA = 1


class binaryReader(object):
    def __init__(self, filename):
        
        if not os.path.isfile(filename):
            print(('Filename',filename,'does not exist'))
            return;
            
        self.__filename=filename;
        self.__f = None;
        self.__isdebug=0;

        self.__open()
        
        return;
    
    def __open(self):
        self.__f=open(self.__filename,'rb')
        return

    def close(self):
        self.__f.close()
        self.__f=None;
        return

    def __del__(self): 
        '''
        dtor
        '''
        if self.__f:
            self.close()
        return

    def get_filename(self):
        return self.__filename

    def get_handler(self):
        return self.__f
    
    def set_debug(self, code=Debug.PLOT_DATA):
        '''
        code = 0 : no debug
        code = 1 : update update plot every time 
        cpde = 2 : 
        '''
        self.__isdebug=code;
        return 

    def get_debug(self):
        return self.__isdebug
    
    def get(self):
        '''
        get method virtual interface
        '''
        pass

    def __read_data(self, nsamples, form, chunk_size):

        event=list();
        for i in range(0, int(nsamples)):
            event.append(struct.unpack(form, self.get_handler().read(chunk_size))[0])
            
        return event 
    
    
class wavedumpReader(binaryReader):
    def __init__(self, filename):
        super().__init__(filename)

        return
    

    def get(self):
        
        ev=dict()
        
        '''
        The HEADER is so composed:
        <header0> (32 bit --> 4 bytes) EventSize (i.e. header + samples)
        <header1> (32 bit --> 4 bytes) Board ID
        <header2> (32 bit --> 4 bytes) Pattern (meaningful only for VME boards)
        <header3> (32 bit --> 4 bytes) Channel
        <header4> (32 bit --> 4 bytes) Event Counter
        <header5> (32 bit --> 4 bytes) Trigger Time Tag
        '''

        # Read header
        chunk_size = 4; # --> 4 bytes 
        for label in ['size', 'boardid', 'pattern', 'ch', 'counter', 'ttag']:
            bread = self.get_handler().read(chunk_size);
            if not bread: return None 
            ev.update({label:struct.unpack('I', bread)[0]})

        ev.update({'evsize': int((ev['size']-(6*chunk_size))/2)})

        # Read data
        ev.update({'trace': self._binaryReader__read_data(int(ev['evsize']), 'H', 2)})
        
        # Plot data
        if self.get_debug() is not Debug.NO_DEBUG and 'trace' in ev:
            plot_trace(ev['trace'])
            
        return ev;


    
class compassReader(binaryReader):
    def __init__(self, filename, calibrated=False):
        super().__init__(filename)

        self.__iscal=bool(calibrated)

        return
    
    
    def get(self):
        
        ev=dict()
        
        '''
        The HEADER is so composed:
        <header0>  (16 bit --> 2 bytes) Board ID (int)
        <header1>  (16 bit --> 2 bytes) Channel (int)
        <header2>  (64 bit --> 8 bytes) Timestamp in ps (int)
        <header3>  (16 bit --> 2 bytes) Energy in channel (int)
        <header3b> (64 bit --> 8 bytes) Energy in keV/MeV (double)
        <header4>  (16 bit --> 2 bytes) Energy short (int)
        <header5>  (32 bit --> 4 bytes) Flag (bit-by-bit, 32 bit)
        <header6>  (32 bit --> 4 bytes) Number of Wave samples to be read (int)
        

        Interpret bytes as packed binary data
        https://docs.python.org/3/library/struct.html
        '''

        # Read header
        label      = ['board', 'channel', 'ttag', 'pulseheight', 'flag' , 'evsize'] if not self.__iscal else \
                     ['board', 'channel', 'ttag', 'pulseheight', 'energy', 'flag' , 'evsize']
        chuck_size = [2, 2, 8, 2, 4, 4] if not self.__iscal else \
                     [2, 2, 8, 2, 8, 4, 4] 
        form       = ['H', 'H' , 'Q', 'H', 'I', 'I'] if not self.__iscal else \
                     ['H', 'H' , 'Q', 'H', 'd', 'I', 'I']
        for l, c, f in zip(label, chuck_size, form):
            bread = self.get_handler().read(c);
            if not bread: return None 
            ev.update({l:struct.unpack(f, bread)[0]})
            
        # Read data
        ev.update({'trace': self._binaryReader__read_data(int(ev['evsize']), 'H', 2)})

        # Plot data
        if self.get_debug() is not Debug.NO_DEBUG and 'trace' in ev:
            plot_trace(ev['trace'])

        return ev


def get_compass_size(filename,calibrated=True):
    if not os.path.isfile(filename):
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
            
    fh=open(filename,'rb')
    
    label      = ['board', 'channel', 'ttag', 'pulseheight', 'flag' , 'evsize'] if not calibrated else \
                 ['board', 'channel', 'ttag', 'pulseheight', 'energy', 'flag' , 'evsize']
    chuck_size = [2, 2, 8, 2, 4, 4] if not calibrated else \
                 [2, 2, 8, 2, 8, 4, 4] 
    form       = ['H', 'H' , 'Q', 'H', 'I', 'I'] if not calibrated else \
                 ['H', 'H' , 'Q', 'H', 'd', 'I', 'I']
    
    header=dict()
                 
    for l, c, f in zip(label, chuck_size, form):
        bread = fh.read(c);
        if not bread: return None 
        header.update({l:struct.unpack(f, bread)[0]})
    
    fh.close() 
    evsize=header['evsize']
    file_size=os.path.getsize(filename)
       
    return int(evsize),int(file_size/(2*evsize+sum(chuck_size)))
                    
def plot_trace(trace):

    plt.ion();
    plt.clf()
    plt.plot(trace)
    plt.xlabel(r'Time [Samples]')
    plt.ylabel(r'Amplitude [ADC count]')
    plt.grid(True)
    plt.show()

    
    return 
