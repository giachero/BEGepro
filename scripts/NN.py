#!/usr/bin/env python

from begepro.rw import CAENhandler_new  as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

def main():

    print('\n')
    print('\n')
    
    filename='/home/marco/work/tesi/data/Std-232Th-3Bq-AEcalibration-im010421/Std-232Th-3Bq-AEcalibration-im010421.npy'
    coll=ca.NPYreader(filename,True).get_event()
    matrix=coll.get_parameters()
    
    df=pd.DataFrame(matrix,columns=['index']+list(coll.get_dict().keys())[2:-1])
    print(df)
    
    df['labels']=(df['ae']>1.6e-2).astype(int)
    print(df)
    
    data=df[['risetime','n_peaks','labels']]
    print(data)
    
    #Separate sets of data
    
    n_training=0.8
    n_validation=0.1
    n_test=0.1 
    
    #I separate MSE from SSE cutting ae on 1.6e-2
        
    train,test=train_test_split(data,test_size=0.2,random_state=11)
    test,val=train_test_split(test,test_size=0.5,random_state=11)
    
    #NN
        
    model = keras.Sequential(
        [
            layers.Dense(2, activation="relu", name="layer1",input_shape=(len(data.columns)-1,)),
            layers.Dense(3, activation="relu", name="layer2"),
            layers.Dense(4, activation="relu", name="layer3"),
            layers.Dense(1, activation='sigmoid', name="layer4"),
        ]
    )
    
    model.summary()
    
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    history=model.fit(train[['risetime','n_peaks']],train['labels'], batch_size=516,epochs=100,validation_data=(val[['risetime','n_peaks']],val['labels']))
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    print('aaa')

    
    
        
    return
    
if __name__ == '__main__':
    main()
    
