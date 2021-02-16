import h5py
import numpy as np

class H5Writer(object):
    def __init__(self, labels, filename):
        
        self.__filename = filename
        self.__data     = dict()
        self.__dtype    = dict()
        
        # paramters
        self.__compression=4
        self.__datacnt = 0
        self.__dlen    = 0
        self.__bsize   = 1000
        
        for l in labels:
            self.__data.setdefault(l, [])
            
        return

    def set_data_lenght(self, dlen):
        self.__dlen=int(abs(dlen))
        return
    
    def set_buffer_size(self, bsize=1000):
        self.__buffer_size = int(abs(bsize))
        return 

    def set_compression(self, level):
        self.__compression = int(abs(level) if abs(level) < 10 else 9)
        return 

    def set_data_type(self, d, val=None):
        self.__dtype.update(d if isinstance(d, dict) else {d:val})
        return

    def get_data_type(self):
        return self.__dtype
    
    def add_data(self, d):
        for l in d:
            self.__data[l].append(d[l])
            if self.__datacnt == 0 and l == 'trace':
                self.__dlen=len(self.__data[l][0])
                
        self.__datacnt += 1        
        
        if not  self.__datacnt % self.__buffer_size:
            self.__write_data()
            for l in d:
                self.__data[l]=list()
                
        return 
    
    def __write_data(self):
        with h5py.File(self.__filename,'a') as f:
            for key in self.__data: 
                if key not in list(f.keys()):
                    # Create variable if not exist
                    create_empty_data(f, key,
                                      1 if key != 'trace' else self.__dlen,
                                      self.__dtype[key], compression=self.__compression)
                    
                # Add data to hdf5 file
                self.__data[key]=np.array(self.__data[key])    
                if key == 'trace':
                    self.__data[key][:,1:]=np.diff(self.__data[key],1)
                append_data(f, key, self.__data[key])
                ## data[:,1:]=np.diff(data,1)
                    
        return
    
    def __del__(self): 
        
        
        return


def create_empty_data(hand, key, dim, dt, compression=4):
    '''
    This function creates an empty dataset with a specified size
    '''
    if key not in list(hand.keys()):
        hand.create_dataset(key, (0,dim), maxshape=(None,dim), dtype=dt,
                            shuffle=True, compression="gzip", compression_opts=compression, chunks=True)
    
    return


def append_data(hand, key, data):
    '''
    This function appends data on an existing dataset
    '''
    if key in hand.keys():
        hand[key].resize(hand[key].shape[0]+data.shape[0], 0)
        hand[key][-data.shape[0]:]=data.reshape(-1, 1 if data.ndim ==1 else data.shape[1])
        #print (hand[key].shape)
        
    return
