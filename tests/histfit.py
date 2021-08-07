import numpy as np
import matplotlib.pyplot as plt
import inspect
import math
from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy.signal import find_peaks, peak_widths

import functions

def get_fnc_args(fnc):
    ret = inspect.getfullargspec(fnc)[0]
    ret.remove('x')
    return ret

def get_nonzero_xy(xdata, ydata, xlim):
    interval = np.nonzero((xdata>=xlim[0]) & (xdata<=xlim[1]))[0]
    nonzero_y_pos = np.nonzero(ydata[interval])[0]
    return xdata[interval][nonzero_y_pos], ydata[interval][nonzero_y_pos]


class HistogramFitter (object):
    
    def __init__(self, countsperbin, binedges):
        self.__x = binedges[:-1] + np.diff(binedges)/2
        self.__y = countsperbin
        
        self.__f = None
        self.__xlim = self.__x[0], self.__x[-1]
        self.__pars = dict()
        return
    
    def set_model(self, f, xlim=None, initpars=None):
        if isinstance(f, tuple): 
            self.__f = functions.peakfnc[f[0]][f[1]]
        else:
            self.__f = f
        
        if xlim is not None:
            self.__xlim = xlim
        
        self.__pars.clear()

        if initpars is None:
            for par in get_fnc_args(self.__f):
                self.__pars.setdefault(par, {'name': par, 'init_value': 1})
                
        if isinstance(initpars, dict):
            for par in get_fnc_args(self.__f):
                self.__pars.setdefault(par, {'name': par, 'init_value': initpars[par] if par in initpars else 1})

        if isinstance(initpars, list):
            for pos, par in enumerate(get_fnc_args(self.__f)):
                self.__pars.setdefault(par, {'name': par, 'init_value': initpars[pos] if pos<len(initpars) else 1})
        return
    
    def get_parameters(self):
        return self.__pars
    
    def get_par_keys(self):
        return list(self.__pars)
    
    def get_par_values(self):
        return [self.__pars[par]['init_value'] for par in self.__pars]
    
    def set_par_values(self, par_values):
        if isinstance(par_values, dict):
            for par in par_values:
                if par in self.__pars:
                    self.__pars[par]['init_value'] = par_values[par]
        if isinstance(par_values, list):
            for pos, par in zip(range(len(par_values)), self.__pars):
                self.__pars[par]['init_value'] = par_values[pos]
        return
    
    def get_par_names(self):
           return  [self.__pars[par]['name'] for par in self.__pars]
        
    def set_par_names(self, par_names):
        if isinstance(par_names, dict):
            for par in par_names:
                if par in self.__pars:
                    self.__pars[par]['name'] = par_names[par]
        if isinstance(par_names, list):
            for pos, par in zip(range(len(par_names)), self.__pars):
                self.__pars[par]['name'] = par_names[pos]
        return
    
    def fit(self):
        p0 = [self.__pars[par]['init_value'] for par in self.__pars]

        xdata, ydata = get_nonzero_xy(self.__x, self.__y, self.__xlim)
        eydata = np.sqrt(ydata)
                
        popt, pcov = curve_fit(f              = self.__f,
                               xdata          = xdata,
                               ydata          = ydata,
                               p0             = p0,
                               sigma          = eydata,
                               absolute_sigma = True)
        
        chisq = np.sum(((ydata-self.__f(xdata,*popt))/eydata)**2)
        ndf = len(xdata) - len(p0)
        pval = chi2.sf(chisq, ndf)
        
        ret = {'chisq/ndf' : chisq/ndf,
               'p-value'   : pval,
               'opt'       : dict()}
               
        for pos, par in enumerate(self.__pars):
            self.__pars[par].update({'opt_value'    : popt[pos],
                                     'err_opt_value': np.sqrt(np.diag(pcov))[pos]})
            ret['opt'].update({par: (popt[pos], np.sqrt(np.diag(pcov))[pos])})
        return ret

    def net_counts(self):
        if 'ntail' in self.__pars:
            counts = self.__pars['ngaus']['opt_value'] + self.__pars['ntail']['opt_value']
            error = np.sqrt(self.__pars['ngaus']['err_opt_value']**2 + self.__pars['ntail']['err_opt_value']**2)
            return counts, error
        else:
            return self.__pars['ngaus']['opt_value'], self.__pars['ngaus']['err_opt_value']

    def peak_width(self, height=0.5, xlim=None):
        if xlim is None:
            xlim = self.__xlim
        interval = np.nonzero((xlim[0] <= self.__x) & (self.__x <= xlim[1]))[0]
        peaks, properties = find_peaks(self.__y[interval], prominence=100, width=[0.5])
        if  peaks.shape[0] == 1:
            width = peak_widths(self.__y[interval], peaks, rel_height=(1-height))
            binwidth = np.diff(self.__x[interval])
            ret = (self.__x[interval][int(math.modf(width[3])[1])] + math.modf(width[3])[0]*binwidth[int(math.modf(width[3])[1])]) \
                - (self.__x[interval][int(math.modf(width[2])[1])] + math.modf(width[2])[0]*binwidth[int(math.modf(width[2])[1])])
            return ret
        else:
            print('nope')
            return

    def plot_fit(self, xlim=None, nop=1000):
        if xlim is None:
            xlim = self.__xlim
        popt = [self.__pars[par]['opt_value'] for par in self.__pars]
        xfnc = np.linspace(xlim[0], xlim[1], num=nop)
        plt.plot(xfnc, self.__f(xfnc, *popt), label='fit')
        plt.legend()
        return
    
    def plot_info(self, xlim=None, nop=1000):
        if xlim is None:
            xlim = self.__xlim
        popt = [self.__pars[par]['opt_value'] for par in self.__pars]
        xfnc = np.linspace(xlim[0], xlim[1], num=nop)
        yfnc = self.__f(xfnc, *popt)
        return {'x': xfnc, 'y': yfnc}

    def plot_components(self, xlim=None, nop=1000):
        if xlim is None:
            xlim = self.__xlim
        xfnc = np.linspace(xlim[0], xlim[1], num=nop)
        plt.plot(xfnc, functions.gaus(xfnc, 
                                      self.__pars['ngaus']['opt_value'], 
                                      self.__pars['mu']['opt_value'], 
                                      self.__pars['sigma']['opt_value']),
                 linestyle='--', linewidth=1, label='gaus')
        if 'ntail' in self.__pars: 
            plt.plot(xfnc, functions.tail(xfnc, self.__pars['mu']['opt_value'], 
                                                self.__pars['sigma']['opt_value'],
                                                self.__pars['ntail']['opt_value'],
                                                self.__pars['ttail']['opt_value']),
                     linestyle='--', linewidth=1, label='tail')
        if 'cstep' in self.__pars: 
            plt.plot(xfnc, functions.step(xfnc, self.__pars['mu']['opt_value'], 
                                                self.__pars['sigma']['opt_value'],
                                                self.__pars['cstep']['opt_value']),
                     linestyle='--', linewidth=1, label='step')
        if 'p0' in self.__pars and 'p1' not in self.__pars:
            plt.plot(xfnc, functions.const(xfnc, self.__pars['p0']['opt_value']), 
                     linestyle='--', linewidth=1, label='bkg')
        if 'p1' in self.__pars and 'p2' not in self.__pars:
            plt.plot(xfnc, functions.lin(xfnc, self.__pars['p0']['opt_value'], 
                                               self.__pars['p1']['opt_value']),
                     linestyle='--', linewidth=1, label='bkg')
        if 'p2' in self.__pars:
            plt.plot(xfnc, functions.quad(xfnc, self.__pars['p0']['opt_value'], 
                                                self.__pars['p1']['opt_value'],
                                                self.__pars['p2']['opt_value']),
                     linestyle='--', linewidth=1, label='bkg')
        plt.legend()
        return

        