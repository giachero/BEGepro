import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2


def get_fnc_args(fnc):
    import inspect
    return inspect.getfullargspec(fnc)[0]

def get_nonzero_xy(xdata, ydata, xmin, xmax):
    interval = np.nonzero((xdata>=xmin) & (xdata<=xmax))[0]
    nonzero_y_pos = np.nonzero(ydata[interval])[0]
    return xdata[interval][nonzero_y_pos], ydata[interval][nonzero_y_pos]


class HistogramFitter (object):
    
    def __init__(self, countsperbin, binedges):
        
        self.__x = binedges[:-1] + np.diff(binedges)/2
        self.__y = countsperbin
        
        self.__f = None
        self.__xlim = self.__x[0], self.__x[-1]
        self.__pars = {}
        
        return
    
    def set_model(self, f, xlim=None, initpars=None):
        
        self.__f = f
        
        if xlim is not None: 
            self.__xlim = xlim
        
        self.__pars.clear()
        
        for par in get_fnc_args(self.__f):
            if par != 'x':
                self.__pars.setdefault(par, {'name': par})
                if initpars is not None and par in initpars:
                    self.__pars[par].update({'value': initpars[par]})
                else:
                    self.__pars[par].update({'value': 1})
        return
    
    def get_parameters(self):
        return self.__pars
    
    def get_par_keys(self):
        return list(self.__pars.keys())
    
    def get_par_values(self):
        return [self.__pars[par]['value'] for par in self.__pars]
    
    def set_par_values(self, par_values):
        for par in par_values:
            if par in self.__pars:
                self.__pars[par]['value'] = par_values[par]    
        return
    
    def get_par_names(self):
           return  [self.__pars[par]['name'] for par in self.__pars]
        
    def set_par_names(self, par_names):
        for par in par_names:
            if par in self.__pars:
                self.__pars[par]['name'] = par_names[par]
        return
    
    def fit(self):
        
        p0 = [self.__pars[par]['value'] for par in self.__pars]

        xdata, ydata = get_nonzero_xy(self.__x, self.__y, self.__xlim)
        eydata = np.sqrt(ydata)
                
        popt, pcov = opt.curve_fit(f              = self.__f,
                                   xdata          = xdata,
                                   ydata          = ydata,
                                   p0             = p0,
                                   sigma          = eydata,
                                   absolute_sigma = True)
        
        chisq = np.sum(((ydata-self.__f(xdata,*popt))/eydata)**2)
        ndf = len(xdata) - len(p0)
        prob = chi2.sf(chisq, ndf)
        
        ret = {'chisq' : chisq,
               'ndf'   : ndf,
               'prob'  : prob,
               'opt'   : {}}
               
        for i, par in enumerate(self.__pars):
            self.__pars[par].update({'opt_value'    : popt[i],
                                     'err_opt_value': np.sqrt(np.diag(pcov))[i]})
            ret['opt'].update({self.__pars[par]['name']: (popt[i], np.sqrt(np.diag(pcov))[i])})
            
        return ret
    
    def basicplot(self):
        popt = [self.__pars[par]['opt_value'] for par in self.__pars]
        xfnc = np.linspace(self.__xlim[0], self.__xlim[1], num=1000)
        plt.plot(xfnc, self.__f(xfnc, *popt))
        return
    
    def infoplot(self, xlim=None, nop=1000):
        if xlim is None:
            xlim = self.__xlim
        popt = [self.__pars[par]['opt_value'] for par in self.__pars]
        xfnc = np.linspace(xlim[0], xlim[1], num=nop)
        yfnc = self.__f(xfnc, *popt)
        return {'x': xfnc, 'y': yfnc}