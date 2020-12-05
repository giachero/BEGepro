import numpy as np
import scipy.signal as sgn
import scipy.constants as cnt
import math


def moving_average(array, window, n=1):
    ker = (1/window) * sgn.windows.boxcar(window)
    ret = sgn.convolve(array, ker)
    if n > 1:
        for i in range(n-1): ret = sgn.convolve(ret, ker)
    return ret[n*window:array.shape[0]]

def gaussian_filter(array, sigma):
    ker = (1 / (math.sqrt(2*cnt.pi)*sigma)) * sgn.windows.gaussian(10*sigma, sigma)
    return sgn.convolve(array, ker)

def delayed_diff(array, delta):
    return (np.pad(array, (delta,0), mode='constant') - np.pad(array, (0,delta), mode='constant'))[delta:-delta]