#!/usr/bin/env python

from begepro.rw import CAENhandler_new  as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

import psutil
import random

def main():

    print('\n')
    print('\n')
       
    for i in range(5,50,5):
        filename='/home/marco/work/tesi/data/228Th-grafico-tesi-im260421_1/AnalisiBig3/228Th-grafico-tesi-im260421_1__'+str(i)+'.npy'
        coll=ca.NPYreader(filename,False).get_event()

        if(i==5):
            coll_tot=coll
        else:
            coll_tot=coll_tot+coll
            del(coll)
            
        print('opened '+str(i)+' , Ram: '+str(psutil.virtual_memory()[2]))    
    
    coll_SSE=((coll_tot.subset('energy',0,1618)+coll_tot.subset('energy',1626,2098)+coll_tot.subset('energy',2110,2608)+coll_tot.subset('energy',2618)).subset('ae',1.90e-2,1.95e-2)).subset('energy',1550)
    coll_MSE=((coll_tot.subset('energy',0,1590)+coll_tot.subset('energy',1597)).subset('ae',0,1.6e-2)).subset('energy',1550)
    
    print('Total events: '+str(coll_SSE.n_trace+coll_MSE.n_trace))
    print('MSE: '+str(coll_MSE.n_trace))
    print('SSE: '+str(coll_SSE.n_trace))
    
    #Sets of MSE and SSE don't have the same dimension, so i select randomly some events from the bigger one (SSE)
    indexes=[]    
    random.seed(10)
    for i in range(0,len(coll_SSE.get_energies())):
        i=random.randint(0,coll_MSE.get_energies().size)
        indexes.append(i)
    
        
    coll_MSE=coll_MSE.subset('ae',index=indexes)
    
    coll_tot2=coll_MSE+coll_SSE
    
    matrix=coll_tot2.get_parameters()
    
    print(coll_SSE.get_energies().size/coll_tot2.get_energies().size)
    print(coll_MSE.get_energies().size/coll_tot2.get_energies().size)
    
    del(coll_SSE)
    del(coll_MSE)
    
    df=pd.DataFrame(matrix,columns=['index']+list(coll_tot2.get_dict().keys())[2:-2]) 
    print(df)
    
    df['labels']=(df['ae']>1.7e-2).astype(int) #1 for SSE and 0 for MSE

    print(df)
    
    
    
    data=df[['risetime','n_peaks','zeros_2der','n_peaks_2der','simm','labels']]
    print(data)
       
    #Split of data set:    train=0.72 ,test=0.09, validation=0.09, evaluation=0.1
        
    a,b=train_test_split(data,test_size=0.9,random_state=11) #a=0.1 b=0.9
    
    train,test=train_test_split(b,test_size=0.2,random_state=11) #train=0.9*0.8=0.72   test=0.9*0.2=0.18
    
    test,val=train_test_split(test,test_size=0.5,random_state=11)#test=0.18*0.5=0.09     val=0.18*0.5=0.09
    

    
    #NN
        
    model = keras.Sequential(
        [
            layers.Dense(5, activation="relu", name="layer1",input_shape=(len(data.columns)-1,)),
            layers.Dense(4, activation="relu", name="layer2"),
            layers.Dense(3, activation="relu", name="layer3"),
            layers.Dense(1, activation='sigmoid', name="layer4"),
        ]
    )
    
    model.summary()
    
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    history=model.fit(train[['risetime','n_peaks','zeros_2der','n_peaks_2der','simm']],train['labels'],batch_size=516,epochs=500,
                      validation_data=(val[['risetime','n_peaks','zeros_2der','n_peaks_2der','simm']],val['labels']))
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    print('aaa')
    
    loss,acc=model.evaluate(a[['risetime','n_peaks','zeros_2der','n_peaks_2der','simm']],a['labels'],verbose=2)
    
    print("Model, accuracy: {:5.2f}%".format(100 * acc))
    
    #model.save('/home/marco/work/tesi/BEGepro/scripts/Model_228Th_with_diff')

    
    
        
    return
    
if __name__ == '__main__':
    main()
    
