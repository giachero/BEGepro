from begepro.autoencoder.sn_analysis import Comparison, compute_threshold
from begepro.autoencoder.dataloading import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from begepro.autoencoder.autoencoder_classifier import Autoencoder, Classifier

opts = {"loadpath": "/mnt/e/BEGE_data/waveforms_crioconite", # where .bin files are stored
"savepath": "../../dataset/crioconite", # where .npy files will be saved with readfiles() and loaded with load()
"subsampling_factor": 3}

dloader = DataLoader(opts)
waveforms, _, energies, amplitudes, pulse_height = dloader.load()

# opts_2 = {"loadpath": "/mnt/e/BEGE_data/waveforms_crioconite", # where .bin files are stored
# "savepath": "../../dataset/Th232_2", # where .npy files will be saved with readfiles() and loaded with load()
# "subsampling_factor": 3}

# dloader_2 = DataLoader(opts_2)
# waveforms_2, _,energies_2, amplitudes_2, pulse_height_2 = dloader_2.load()

# dataset = Dataset(waveforms, energies, amplitudes, pulse_height)

# energies = np.concatenate([energies, energies_2])
# waveforms = np.concatenate([waveforms, waveforms_2])
# amplitudes = np.concatenate([amplitudes, amplitudes_2])
# pulse_height = np.concatenate([pulse_height, pulse_height_2])

# dataset = Dataset(waveforms, energies, amplitudes, pulse_height)
# train, val, test = dataset.train_val_test_split(0.0509, 0.254, 0.6951, "../../../BEGe/dataset/Th232_splitting_map")

dense_list = [50, 35, 5, 35, 50] # neurons for each hidden layers of the autoencoder.
lrelu_slopes = [0.05, 0.05, 0.01, 0.05, 0.05] # alpha parameter for the LeakyReLU activation function.
autoencoder = Autoencoder(dense_list, lrelu_slopes) 
autoencoder.load_model("../../../BEGe/models/autoencoder", "../../../BEGe/models/encoder")

M_ELECTRON = 511

# # Training regions must be defined as list [energy_centre, energy_std].
# # .get_classification_sample will select the energy range [energy_centre - 2*energy_std, energy_centre + 2*energy_std]


# # MSE regions:
# region_Tlpeak = [2615, 3]
# region_Tlsep = [2615 - 1*M_ELECTRON, 3]
# region_train_mcompton = [2500, 32]

# # SSE regions:
# region_dep = [2615 - 2*M_ELECTRON, 3]
# region_cedge = [2250, 60]

# region_mse = [region_Tlpeak, region_Tlsep, region_train_mcompton]
# region_sse = [region_dep, region_cedge]

# wf_train, wf_val, label_train, label_val = dataset.get_classification_sample(train, val, region_mse, region_sse)


# encoded_val, mus, stds = autoencoder.encode(wf_val)
# encoded_training, _, _ = autoencoder.encode(wf_train, mus, stds)
mus = np.array([2.9214985, 3.264203 , 3.065095 , 2.987435 , 2.1484957])
stds = np.array([0.42397776, 0.42308047, 1.5442389 , 0.74881005, 0.252582  ])
# dense_list = [10, 10]
# lrelu_slopes = [0.05, 0.05]

import pdb; pdb.set_trace()
classifier = Classifier(dense_list = dense_list, lrelu_slopes = lrelu_slopes, input_size = 5)
classifier.load_model("../../../BEGe/models/classifier")

test_encoded, _, _ = autoencoder.encode(waveforms, mus, stds)
test_scores = classifier.classifier.predict(test_encoded)
