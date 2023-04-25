""" This module contains utilities for estimating peak S/B ratios and comparing the performances of the A/E method and the autoencoder + NN one."""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings

def sb_fit_function(x, a, b, mu, sigma, c):
    return a * np.exp(-(x-mu)**2/(2*sigma**2)) + b + c*(x-mu)

def compute_threshold(predictor, num_saved, model, stress_limit = 100000):
    a = -0.0001
    b = np.max(predictor) + 0.001
    stress = 0
    if model == "avse":
        while True:
            stress = stress+1
            c = 0.5*(a+b)
            n = np.sum(predictor < c)
            if n > num_saved:
                b = c
            elif n < num_saved:
                a = c
            elif num_saved == n:
                break
            if stress > stress_limit and np.abs(num_saved - n) < 2:
                break
            elif stress > stress_limit:
                warnings.warn(f"Correct matching between score and A/E cuts not found. Discrepancy: {np.abs(num_saved - n)}") 
                break
    elif model == "nn":
        while True:
            stress = stress+1
            c = 0.5*(a+b)
            n = np.sum(predictor > c)
            if n < num_saved:
                b = c
            elif n > num_saved:
                a = c
            elif num_saved == n:
                break

            if stress > stress_limit and np.abs(num_saved - n) < 2:
                break
            elif stress > stress_limit:
                warnings.warn(f"Correct matching between score and A/E cuts not found. Discrepancy: {np.abs(num_saved - n)}") 
                break
    return c

def exclude_unwanted_peaks(xdata, peaks, properties, mu):
    fit_mask = np.ones(xdata.shape[0])
    mu_location_idx = np.argmin(np.abs(xdata[peaks] - mu))
    for i, p in enumerate(peaks):
        if np.logical_and(i != mu_location_idx, np.abs(xdata[peaks[i]] - mu) > 2):
            unwanted_peak_location_idx = peaks[i]
            width = np.round(1*properties['widths'][i]).astype(int)
            fit_mask[unwanted_peak_location_idx] = 0
            for j in range(1, width+1):
                if unwanted_peak_location_idx-j >= 0:
                    fit_mask[unwanted_peak_location_idx-j] = 0
                if unwanted_peak_location_idx+j < xdata.shape[0]:
                    fit_mask[unwanted_peak_location_idx+j] = 0
    return fit_mask.astype(bool)

def compute_sn(data, selection, mu_true, energy_mask, width, nbins):
    masked_energies = data[energy_mask]
    energies = masked_energies[selection]
    hist = np.histogram(energies, bins = nbins, range = [mu_true -width, mu_true +width])
    ydata = hist[0]
    xdata = (hist[1][:-1] +  hist[1][1:])/2 
    sigma = np.sqrt(ydata)/ydata.sum()
    sigma[sigma == 0] = 1
    ydata = ydata/ydata.sum()
    prominence = 0.005

    while True:
        peaks, properties = find_peaks(ydata, prominence=prominence, width=1.5)
        if len(peaks) > 0:
            break
        else:
            prominence = prominence - 0.0005
        if prominence < 0:
            warnings.warn("No peaks found.")
            return 0, 0
    a0 = 1
    b0 = 0
    mu0 = xdata[peaks[np.argmin(np.abs(xdata[peaks] - mu_true))]]
    sigma0 = 3
    c0 = 0

    fit_mask = exclude_unwanted_peaks(xdata, peaks, properties, mu0)
    xdata = xdata[fit_mask]
    ydata = ydata[fit_mask]
    sigma = sigma[fit_mask]
    xdata = xdata[3:-3]
    ydata = ydata[3:-3]

    sigma = np.sqrt(ydata)
    sigma[sigma == 0] = 1
    popt, pcov = curve_fit(sb_fit_function, xdata, ydata, p0 = [a0, b0, mu0, sigma0, c0], sigma = sigma, maxfev = 100000, bounds=((0, 0, 0, 0, -1e5), (1e5, 1e5, 3e3, 20, 1)), method = "trf")

    stds = np.sqrt(np.diag(pcov))
    bkg = popt[1]
    std_b = stds[1]
    sig = popt[0]
    std_s = stds[0]

    sb = sig/bkg
    std_sb = np.sqrt((std_s/bkg)**2 + (sig*std_b/bkg**2)**2 - sig/bkg**3*pcov[0, 1])

    if std_sb > sb:
        warnings.warn("The fit did not converge.")
        return None, None
    else:
        return sb, std_sb

class Comparison():
    def __init__(self, data, avse, scores):
        self.data = data
        self.avse = avse
        self.scores = scores
    
    def compare(self, peaklist, scan, benchmark, width = 30, nbins = 90):
        # width = 30
        # nbins = 90
        sb_nn = np.zeros((len(scan), len(peaklist)))
        std_sb_nn = np.zeros((len(scan), len(peaklist)))
        sb_avse = np.zeros((len(scan), len(peaklist)))
        std_sb_avse = np.zeros((len(scan), len(peaklist)))

        nonbenchmark_threshold = np.zeros((len(scan), len(peaklist)))

        for j, peak in enumerate(peaklist):
            print(f"Estimating S/B for peak at {peak} keV")
            energy_mask = np.logical_and(self.data > peak - width, self.data < peak + width)
            masked_scores = self.scores[energy_mask]
            masked_avse = self.avse[energy_mask]
            for i, cut in enumerate(scan):
                if benchmark == "nn":
                    predictions_nn = masked_scores > cut
                    threshold = compute_threshold(masked_avse, predictions_nn.sum(), "avse")
                    predictions_avse = masked_avse < threshold
                elif benchmark == "avse":
                    predictions_avse = masked_avse < cut
                    threshold = compute_threshold(masked_scores, predictions_avse.sum(), "nn")
                    predictions_nn = masked_scores > threshold
                nonbenchmark_threshold[i,j] = threshold
                sb_nn[i, j], std_sb_nn[i, j] = compute_sn(self.data, predictions_nn, peak, energy_mask, width, nbins)
                sb_avse[i,j], std_sb_avse[i, j] = compute_sn(self.data, predictions_avse, peak, energy_mask, width, nbins)

        return sb_nn, std_sb_nn, sb_avse, std_sb_avse, nonbenchmark_threshold