import os, errno 
import struct
import pylab as plt

from lxml import etree as ET

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


    
'''
enum style class
'''
class Debug(object):
    NO_DEBUG  = 0
    PLOT_DATA = 1

    

class wavedumpReader(object):
    def __init__(self,filename):
        
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
    
    def set_debug(self, code=2):
        '''
        code = 0 : no debug
        code = 1 : update plot every 1000 events
        cpde = 2 : update plot every time 
        '''
        self.__isdebug=code;
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
        chunk_size = 4; # --> 4 bytes 
        for label in ['size', 'boardid', 'pattern', 'ch', 'counter', 'ttag']:
            bread = self.__f.read(chunk_size);
            if not bread: return None 
            ev.update({label:struct.unpack('I', bread)[0]})


        ev.update({'evsize':int((ev['size']-(6*chunk_size))/2)})

        '''
        
        '''
        event=list();
        chunk_size = 2; 
        for i in range(0, int(ev['evsize'])):
            event.append(struct.unpack('H', self.__f.read(chunk_size))[0])

        ev.update({'trace':event})
        
        if self.__isdebug is not Debug.NO_DEBUG:
            plt.ion();
            plt.clf()
            plt.plot(ev['trace'])
            plt.xlabel(r'Time [Samples]')
            plt.ylabel(r'Amplitude [ADC count]')
            plt.grid(True)
            plt.show()

            
        return ev;
