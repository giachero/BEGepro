def moving_average(array, window, n=1):
    ker = (1/window) * sgn.windows.boxcar(window)
    ret = sgn.convolve(array, ker)
    if n > 1:
        for i in range(n-1): ret = sgn.convolve(ret, ker)
    return ret

def gaussian_filter(array, sigma):
    ker = (1 / (math.sqrt(2*cnt.pi)*sigma)) * sgn.windows.gaussian(10*sigma, sigma)
    ret = sgn.convolve(array, ker)
    return ret

def delayed_diff(array, delta):
    arr1 = np.pad(array, (delta,0))
    arr2 = np.pad(array, (0,delta))
    ret = arr2 - arr1
    return ret[:-delta]