import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
from scipy.stats import chi2
from sys.stdout import write


def lin(p, x):
    return p[0] + p[1]*x

def quad(p, x):
    return p[0] + p[1]*x + p[2]*x**2 

def cube(p, x):
    return p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3

def quart(p, x):
    return p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4


class EnergyCalibration(object):
    
    def __init__(self, pulseheights):
        
        self.__pulseheights = pulseheights
        
        self.__stden = dict()
        with open('../en_res/calibration_energies.json', 'r') as f:
            self.__stden = json.load(f)
        
        self.__fnc = quad
        self.__pars = [1.,1.,1.]
        
        self.__x = list()
        self.__y = list()
        self.__ex = list()
        self.__ey = list()
        
        for isotope in self.__pulseheights:
            if isotope in self.__stden:
                self.__x.append(self.__pulseheights[isotope][0])
                self.__y.append(self.__stden[isotope][0])
                self.__ex.append(self.__pulseheights[isotope][1])
                self.__ey.append(self.__stden[isotope][1])
            else:
                write('No standard energy found for {isotope} --- I will not use it'.format(isotope=isotope))  
        
        return

    def set_model(self, fnc, initpars=None):
        self.__fnc = fnc
        if initpars is not None:
            self.__pars = initpars
        return
            
    def set_parameters(self, initpars):
        self.__pars = initpars
        return
        
    def get_parameters(self):
        return self.__pars
        
    def calibrate(self):
        
        data = odr.RealData(x  = self.__x,
                            y  = self.__y,
                            sx = self.__ex,
                            sy = self.__ey)
        
        model = odr.Model(self.__fnc)
        
        cal_odr = odr.ODR(data  = data,
                          model = model,
                          beta0 = self.__pars)

        output = cal_odr.run()
             
        chisq = output.sum_square
        ndf = round(output.sum_square/output.res_var)
        prob = chi2.sf(chisq, ndf)
        
        self.__pars = output.beta
        
        ret = {'chisq': chisq,
               'ndf'  : ndf,
               'prob' : prob,
               'opt'  : {}}
        
        for i in range(len(output.beta)):
            ret['opt'].update({'p'+str(i): (output.beta[i], output.sd_beta[i])})
        
        return ret
    
    def basicplot(self, xlim=None, nop=1000):
        if xlim is None: 
            xlim = np.min(self.__x), np.max(self.__x)
        xfnc = np.linspace(xlim[0], xlim[1], num=nop)
        plt.errorbar(self.__x, self.__y, xerr = self.__ex, yerr = self.__ey, fmt='.')
        plt.plot(xfnc, self.__fnc(self.__pars, xfnc))
        plt.xlabel('Channel Number')
        plt.ylabel('Energy [keV]')
        plt.title('Energy Calibration')
        return
    
    def infoplot(self, xlim, nop=1000):
        xfnc = np.linspace(xlim[0], xlim[1], num=nop)
        yfnc = self.__fnc(self.__pars, xfnc)
        ret = {'x' : self.__x,
               'y' : self.__y,
               'ex': self.__ex,
               'ey': self.__ey,
               'xf': xfnc,
               'yf': yfnc}
        return ret