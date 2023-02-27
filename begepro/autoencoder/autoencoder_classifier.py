# import tensorflow as tf
# import keras
# import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, Loss
import numpy as np

set_seed(42)

kernel_init = RandomNormal(stddev = 0.01, mean = 0.01, seed = 42)

def standardize_data(distribution, mu = None, std = None):
    if (mu is None) and (std is None):
        mu = np.mean(distribution, axis = 0)
        std = np.std(distribution, axis = 0)
    distribution = (distribution - mu) / std
    return distribution, mu, std

class Autoencoder():
    def __init__(self, dense_list = None, lrelu_slopes = None, input_size = 122):
        self.dense_list = dense_list
        self.lrelu_slopes = lrelu_slopes
        self.input_size = input_size

    def build_network(self, bottleneck_size = None):
        input_layer = Input(shape = (self.input_size,), name = "input")
        x = input_layer #.output
        bottleneck_idx = np.argmin(self.dense_list)
        for i, (neurons, alpha) in enumerate(zip(self.dense_list, self.lrelu_slopes)):
            if i != bottleneck_idx:
                x = Dense(neurons, activation = LeakyReLU(alpha), kernel_initializer=kernel_init, name = "dense_"+str(i))(x)
            else:
                if bottleneck_size:
                    bottleneck = Dense(bottleneck_size, kernel_initializer=kernel_init, name = "bottleneck")(x)
                    x = LeakyReLU(alpha, name = "bottleneck_activation")(bottleneck)
                else:
                    bottleneck = Dense(neurons, kernel_initializer=kernel_init, name = "bottleneck")(x)
                    x = LeakyReLU(alpha, name = "bottleneck_activation")(bottleneck)
        output_layer = Dense(self.input_size, activation = "sigmoid", name = "output")(x)
        autoencoder = Model(input_layer, output_layer)
        encoder = Model(input_layer, bottleneck)
        optimizer = Adam(learning_rate=0.00035)
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

        self.autoencoder = autoencoder
        self.encoder = encoder 
    
    def train(self, train_data, val_data, epochs = 3000, batch_size = 1120, verbose = 2, save = False, bottleneck_size = None):
        checkpoint_filepath = '/tmp/checkpoint'
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        
        self.build_network(bottleneck_size)
        self.autoencoder.fit(train_data, train_data,
            epochs = epochs,
            batch_size = batch_size,
            shuffle = True,
            validation_data = (val_data, val_data),
            verbose = verbose,
            callbacks = [model_checkpoint_callback])
        self.autoencoder.load_weights(checkpoint_filepath, by_name = False)
        self.encoder = Model(inputs = self.autoencoder.input,
                        outputs = self.autoencoder.get_layer("bottleneck").output)

        if save:
            self.autoencoder.save("../../models/autoencoder")
            self.encoder.save("../../models/encoder")
    
    def train_scan(self, n_range, train_data, val_data, epochs = 3000, batch_size = 1120, verbose = 2):
        n = n_range[1] - n_range[0]
        losses_tr = np.ones(n)
        losses_val = np.ones(n)
        for i in range(n):
            self.train(train_data, val_data, epochs, batch_size, verbose, bottleneck_size=i+n_range[0])
            preds_tr = self.autoencoder.predict(train_data)
            preds_val = self.autoencoder.predict(val_data)
            losses_tr[i] = MeanSquaredError(preds_tr, train_data)
            losses_val[i] = MeanSquaredError(preds_val, val_data)
            np.savetxt("../../results/losses_scan_train", losses_tr)
            np.savetxt("../../results/losses_scan_val", losses_val)
    
    def encode(self, waveforms, mu = None, std = None):
        if not hasattr(self, "encoder"):
            print("The model must be built first")
            return None
        else:
            predictions = self.encoder.predict(waveforms)
            predictions, mus, stds = standardize_data(predictions)
            return predictions, mus, stds

    def load_model(self, path_autoencoder, path_encoder):
        self.autoencoder = load_model(path_autoencoder, compile=False)
        self.encoder = load_model(path_encoder, compile=False)

class CustomAccuracy(Loss):
    def __init__(self):
        super().__init__()
    def call(self, y_true, y_pred):
        bce = BinaryCrossentropy(from_logits=False)
        return bce(y_true, y_pred)#, sample_weight = -y_true + 2)

class Classifier():
    def __init__(self, input_size, dense_list = None, lrelu_slopes = None):
        self.dense_list = dense_list
        self.lrelu_slopes = lrelu_slopes
        self.input_size = input_size
        super().__init__

    def build_network(self):
        input_layer = Input(shape = (self.input_size,), name = "input")
        x = input_layer #.output
        for i, (neurons, alpha) in enumerate(zip(self.dense_list, self.lrelu_slopes)):
            x = Dense(neurons, activation = LeakyReLU(alpha), name = "dense_"+str(i))(x)

        output_layer = Dense(1, activation = "sigmoid", name = "output")(x)
        classifier = Model(input_layer, output_layer)

        optimizer = Adam(learning_rate=0.00035)

        classifier.compile(optimizer=optimizer, loss=CustomAccuracy(), metrics=["accuracy"])

        self.classifier = classifier

    def train(self, train_data, train_labels, val_data, val_labels, epochs = 200, batch_size = 500, verbose = 2, save = False):

        checkpoint_filepath = '/tmp/checkpoint'
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        
        self.build_network()
        self.classifier.fit(train_data, train_labels,
            epochs = epochs,
            batch_size = batch_size,
            shuffle = True,
            validation_data = (val_data, val_labels),
            verbose = verbose,
            callbacks = [model_checkpoint_callback])
        self.classifier.load_weights(checkpoint_filepath, by_name = False)

        if save:
            self.classifier.save("../../models/classifier")
    
    def load_model(self, path_classifier):
        self.classifier = load_model(path_classifier, compile=False)
