#!/usr/bin/env python

from begepro.rw import CAENhandler_new  as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import IPython 

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

model = tf.keras.models.load_model('/home/marco/work/tesi/BEGepro/scripts/Model_228Th_with_diff')
model.summary()

for i in range(55,155,5):
        filename='/home/marco/work/tesi/data/228Th-grafico-tesi-im260421_1/AnalisiBig3/228Th-grafico-tesi-im260421_1__'+str(i)+'.npy'
        coll=ca.NPYreader(filename,False).get_event()
        
        matrix=coll.get_parameters()
        
        
        df=pd.DataFrame(matrix,columns=['index']+list(coll.get_dict().keys())[2:-2])
        df['labels']=(df['ae']>1.7e-2).astype(int) #1 for SSE and 0 for MSE
        data=df[['risetime','n_peaks','zeros_2der','n_peaks_2der','simm','labels']]

        res=model.predict(data[['risetime','n_peaks','zeros_2der','n_peaks_2der','simm']])
        
        np.save('/home/marco/work/tesi/data/labels__'+str(i),res)        
            
        print('evaluated file '+str(i))
        

    


#l=len(data[['risetime','n_peaks','zeros_2der','n_peaks_2der']])

#IPython.embed()

#np.save('/home/marco/work/tesi/data/labels', (res>0.5)*1)

#v=np.load('/home/marco/work/tesi/data/NN/labels.npy')

#print(v)
