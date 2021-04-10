import numpy as np
import scipy.signal as sgn
import scipy.constants as cnt
import math


def moving_average(array, window, n=1):
    ker = (1/window) * sgn.windows.boxcar(window)
    ret = sgn.convolve(array, ker)
    if n > 1:
        for i in range(n-1): ret = sgn.convolve(ret, ker)
    return ret[:array.shape[0]]

def delayed_diff(array, delta):
    return (np.pad(array, (0,delta), mode='constant') - np.pad(array, (delta,0), mode='constant'))[delta:-delta]

def pz_corr(array, tau=11000):
    return (np.pad(array, (0,1)) + np.pad(np.cumsum(array), (1,0))/tau)[1:-1]

def gauss_filter(array, sigma):
    ker = (1 / (math.sqrt(2*cnt.pi)*sigma)) * sgn.windows.gaussian(10*sigma, sigma)
    return sgn.convolve(array, ker)

def gengauss_filter(array, sigma, p):
    ker = (p/(2**(1/(2*p))*sigma*math.gamma(1/(2*p)))) * sgn.windows.general_gaussian(10*sigma, p, sigma)
    return sgn.convolve(array, ker)

def trap_filter(array, rt, ft):
    return (np.pad(moving_average(array, rt), (0, rt+ft)) - np.pad(moving_average(array, rt), (rt+ft, 0)))[:-(rt+ft)]

def curr_filter(array):
    temp = sgn.savgol_filter(array, 19, 2)
    return sgn.savgol_filter(temp, 13, 2, deriv=1)