""" This module contains utility functions for handling the dataset for the autoencoder"""
from begepro.rw import CAENhandler
import numpy as np
import os
from begepro.dspro import filters as flt
from sklearn.model_selection import train_test_split

M_ELECTRON = 511 # Electron mass in keV 

normalize = lambda wave: (wave - wave.min())/(wave.max() - wave.min())

def moving_average(full_wf, subsampling_rate):
    avg_points = np.convolve(full_wf, np.ones(subsampling_rate), 'valid') / subsampling_rate
    subsampled_wf = avg_points[::subsampling_rate]
    return subsampled_wf 

class DataLoader(dict):
    def __init__(self, options):
        self.loadpath = options["loadpath"]
        self.savepath = options["savepath"] 
        self.subsampled_size = 366 // options["subsampling_factor"]

    def readfile(self, rd, n, min_en, max_en, calibrated):  
        waveforms = np.zeros((n, self.subsampled_size))
        currents = np.zeros((n, self.subsampled_size))
        ens = np.zeros(n)
        amps = np.zeros(n)
        pulse_height = np.zeros(n)
        counter = 0
        while True:
            try:
                data = rd.get()
                if calibrated:
                    en = data['energy']
                    p_height = data['pulseheight']
                else:
                    p_height = data['pulseheight']
                    en = p_height*0.20484472 - 0.08026583
                if not(en < max_en and en > min_en):
                    continue
                ens[counter] = en
                pulse_height[counter] = p_height
                trace = np.array(data['trace'])
                curr = flt.curr_filter(trace)
                amps[counter] = np.max(curr) 
                trace = normalize(moving_average(trace, 3))
                curr = normalize(moving_average(curr, 3))
                waveforms[counter] = trace
                currents[counter] = curr
                counter = counter+1
            except:
                break
        if counter < n:
            waveforms = waveforms[:counter]
            currents = currents[:counter]
            ens = ens[:counter]
            amps = amps[:counter]
            pulse_height = pulse_height[:counter]
        return waveforms, currents, ens, amps, pulse_height

    def readfiles(self, n, emin, emax, calibrated = True):
        waveforms = np.zeros((n, self.subsampled_size))
        currs = np.zeros((n, self.subsampled_size))
        ens = np.zeros(n)
        amps = np.zeros(n)
        pulse_height = np.zeros(n)
        counter = 0
        for fname in os.listdir(self.loadpath):
            print("reading", fname)
            rd = CAENhandler.compassReader(self.loadpath + "/" + fname,calibrated=calibrated)
            wf, cu, es, am, ph = self.readfile(rd, n, emin, emax, calibrated)
            dim = es.shape[0]
            print("I found", dim, "waveforms in the chosen energy region.")
            if counter+dim < n:
                waveforms[counter:counter+dim] = wf
                currs[counter:counter+dim] = cu
                ens[counter:counter+dim] = es
                amps[counter:counter+dim] = am 
                pulse_height[counter:counter+dim] = ph
            if counter+dim >= n:
                waveforms[counter:] = wf[:n-counter]
                currs[counter:] = cu[:n-counter]
                ens[counter:counter+dim] = es[:n-counter]
                amps[counter:counter+dim] = am[:n-counter]
                pulse_height[counter:counter+dim] = ph[:n-counter]

            self.save(waveforms, currs, ens, amps, pulse_height)
            if counter + dim >= n:
                break
            counter = counter+ dim
        return waveforms, currs, ens, amps, pulse_height

    def save(self, waveforms, currents, ens, amps, pulse_height):
        zeros_idxs = np.argwhere(ens == 0)
        if zeros_idxs.shape[0]:
            trim_idx = np.min(zeros_idxs)
            waveforms = waveforms[:trim_idx]
            currents = currents[:trim_idx]
            ens = ens[:trim_idx]
            amps = amps[:trim_idx]
            pulse_height = pulse_height[:trim_idx]

        savepath = self.savepath
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        with open(savepath + "/waveforms.npz", "wb") as f:
            np.save(f, waveforms)
        with open(savepath + "/currents.npz", "wb") as f:
            np.save(f, currents)
        with open(savepath + "/energies.npz", "wb") as f:
            np.save(f, ens)
        with open(savepath + "/amplitudes.npz", "wb") as f:
            np.save(f, amps)
        with open(savepath + "/pulse_height.npz", "wb") as f:
            np.save(f, pulse_height)
        print("Files saved in a handy format in ", self.savepath)

    def load(self):
        data_path = self.savepath
        with open(data_path +"/waveforms.npz", "rb") as f:
            waveforms = np.load(f)
        with open(data_path +"/currents.npz", "rb") as f:
            currents = np.load(f)
        with open(data_path +"/amplitudes.npz", "rb") as f:
            amplitudes = np.load(f)
        with open(data_path +"/energies.npz", "rb") as f:
            energies = np.load(f)
        with open(data_path +"/pulse_height.npz", "rb") as f:
            pulse_height = np.load(f)
        return waveforms, currents, energies, amplitudes, pulse_height

#####################################################################

def get_peak_mask(energies, peak_mean, peak_std):
    upper_selection = energies > peak_mean - 2*peak_std
    lower_selection = energies < peak_mean + 2*peak_std
    mask = np.logical_and(upper_selection, lower_selection)
    return mask

def balance_dataset(wf, labels):
    pos = wf[labels == 1]
    neg = wf[labels == 0]
    pos_labels = labels[labels == 1]
    neg_labels = labels[labels == 0]
    if np.sum(pos_labels) / labels.shape[0] > 0.5:
        pos, _, pos_labels, _ = train_test_split(pos, pos_labels, train_size = neg_labels.shape[0])
    else:
        neg, _, neg_labels, _ = train_test_split(neg, neg_labels, train_size = pos_labels.shape[0])
    wf = np.concatenate([pos, neg])
    labels = np.concatenate([pos_labels, neg_labels])
    return wf, labels

class Dataset():
    def __init__(self, waveforms, energies, amplitudes, pulse_height):
        self.wf = waveforms
        self.en = energies
        self.am = amplitudes
        self.ph = pulse_height
    def train_val_test_split(self, train_frac, val_frac, test_frac):
        scalar_data = np.vstack([self.en, self.am, self.ph]).T
        wf_train, wf_val_test, scalar_data_train, scalar_data_val_test = train_test_split(self.wf, scalar_data, train_size = train_frac, test_size = val_frac + test_frac, random_state = 42)

        val_frac = val_frac/(val_frac + test_frac)
        test_frac = 1 - val_frac

        wf_val, wf_test, scalar_data_val, scalar_data_test = train_test_split(wf_val_test, scalar_data_val_test, train_size = val_frac, test_size = test_frac, random_state = 42)

        en_train, am_train, ph_train = scalar_data_train[:,0], scalar_data_train[:,1], scalar_data_train[:,2]
        en_val, am_val, ph_val = scalar_data_val[:,0], scalar_data_val[:,1], scalar_data_val[:,2]
        en_test, am_test, ph_test = scalar_data_test[:,0], scalar_data_test[:,1], scalar_data_test[:,2]

        train = [wf_train, en_train, am_train, ph_train]
        val = [wf_val, en_val, am_val, ph_val]
        test = [wf_test, en_test, am_test, ph_test]
        return train, val, test

    def get_classification_sample(self, data_train, data_val, region_mse, region_sse):
        energies_train = data_train[1]
        energies_val = data_val[1]
        mask_multisite_total_train = np.zeros(energies_train.shape[0])
        mask_multisite_total_val = np.zeros(energies_val.shape[0])
        mask_singlesite_total_train = np.zeros(energies_train.shape[0])
        mask_singlesite_total_val = np.zeros(energies_val.shape[0])  

        for region in region_mse:
            mask_train = get_peak_mask(energies_train, region[0], region[1])
            mask_multisite_total_train = np.logical_or(mask_multisite_total_train, mask_train)
            mask_val = get_peak_mask(energies_val, region[0], region[1])
            mask_multisite_total_val = np.logical_or(mask_multisite_total_val, mask_val)

        for region in region_sse:
            mask_train = get_peak_mask(energies_train, region[0], region[1])
            mask_singlesite_total_train = np.logical_or(mask_singlesite_total_train, mask_train)
            mask_val = get_peak_mask(energies_val, region[0], region[1])
            mask_singlesite_total_val = np.logical_or(mask_singlesite_total_val, mask_val)

        label_train = np.concatenate([np.ones(np.sum(mask_multisite_total_train)), np.zeros(np.sum(mask_singlesite_total_train))])
        label_val = np.concatenate([np.ones(np.sum(mask_multisite_total_val)), np.zeros(np.sum(mask_singlesite_total_val))])

        wf_train = np.concatenate([data_train[0][mask_multisite_total_train], data_train[0][mask_singlesite_total_train] ])
        wf_val = np.concatenate([data_val[0][mask_multisite_total_val], data_val[0][mask_singlesite_total_val] ])

        wf_train, label_train = balance_dataset(wf_train, label_train)
        wf_val, label_val = balance_dataset(wf_val, label_val)

        return wf_train, wf_val, label_train, label_val
        