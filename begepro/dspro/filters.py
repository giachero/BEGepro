import numpy as np
import scipy.signal as sgn
import scipy.constants as cnt
import math

def baseline_sub(array, nsample):
    return array - np.mean(array[0:nsample])

def delayed_diff(array, delta):
    return (np.pad(array, (0,delta), mode='constant') - np.pad(array, (delta,0), mode='constant'))[delta:-delta]

def pz_corr(array, tau):
    return (np.pad(array, (0,1)) + np.pad(np.cumsum(array), (1,0))/tau)[1:-1]

def moving_average(array, window, n=1):
    ker = (1/window) * sgn.windows.boxcar(window)
    ret = sgn.convolve(array, ker)
    if n > 1:
        for i in range(n-1): ret = sgn.convolve(ret, ker)
    return ret[:array.shape[0]]

def gauss_filter(array, sigma, nsample=100, tau=11000):
    baseline_subtracted = baseline_sub(array, nsample=nsample)
    pz_corrected = pz_corr(baseline_subtracted, tau=tau)
    del_differentiated = delayed_diff(pz_corrected, delta=sigma)
    ker = (1 / (math.sqrt(2*cnt.pi)*sigma)) * sgn.windows.gaussian(10*sigma, sigma)
    return sgn.convolve(del_differentiated, ker)

def gengauss_filter(array, sigma, p, nsample=100, tau=11000):
    baseline_subtracted = baseline_sub(array, nsample=nsample)
    pz_corrected = pz_corr(baseline_subtracted, tau=tau)
    del_differentiated = delayed_diff(pz_corrected, delta=sigma)
    ker = (p/(2**(1/(2*p))*sigma*math.gamma(1/(2*p)))) * sgn.windows.general_gaussian(10*sigma, p, sigma)
    return sgn.convolve(del_differentiated, ker)

def trap_filter(array, rt=500, ft=250, nsample=100, tau=11000):
    baseline_subtracted = baseline_sub(array, nsample=nsample)
    pz_corrected = pz_corr(baseline_subtracted, tau=tau)
    return (np.pad(moving_average(pz_corrected, rt), (0, rt+ft)) - np.pad(moving_average(pz_corrected, rt), (rt+ft, 0)))[:-(rt+ft)]

def curr_filter(array, wsmooth=19, dsmooth=2, wdiff=13, ddiff=2):
    temp = sgn.savgol_filter(array, wsmooth, dsmooth)
    return sgn.savgol_filter(temp, wdiff, ddiff, deriv=1)