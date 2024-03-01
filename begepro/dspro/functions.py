import numpy as np
import scipy.constants as cnt
import scipy.special as spc


#POLYNOMIAL FUNCTIONS

def const(x, p0):
    return p0 + 0*x

def lin(x, p0, p1):
    return p0 + p1*x

def quad(x, p0, p1, p2):
    return p0 + p1*x + p2*x**2

def cube(x, p0, p1, p2, p3):
    return p0 + p1*x + p2*x**2 + p3*x**3

def quart(x, p0, p1, p2, p3, p4):
    return p0 + p1*x + p2*x**2 + p3*x**3 + p4*x**4


#PEAK COMPONENTS

def tail(x, mu, sigma, ntail, ttail):
    tail = (ntail/(2.*ttail)) * np.exp((x-mu+sigma**2/(2.*ttail))/ttail) \
    * spc.erfc((x-mu+sigma**2/ttail)/(np.sqrt(2.)*sigma))
    return tail

def step(x, mu, sigma, cstep):
    step = cstep * spc.erfc((x-mu)/(np.sqrt(2.)*sigma))
    return step


#PEAK MODELS

def gaus(x, ngaus, mu, sigma):
    gaus = ngaus / np.sqrt(2. * cnt.pi * sigma ** 2) * np.exp(- 0.5 * ((x - mu) / (sigma)) ** 2)
    return gaus

def gausConst(x, ngaus, mu, sigma, p0):
    #gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    return gaus(x, ngaus, mu, sigma) + p0

def gausLin(x, ngaus, mu, sigma, p0, p1):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    return gaus + p0 + p1*x

def gausQuad(x, ngaus, mu, sigma, p0, p1, p2):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    return gaus + p0 + p1*x + p2*x**2

def gausTail(x, ngaus, mu, sigma, ntail, ttail):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    tail = (ntail/(2.*ttail)) * np.exp((x-mu+sigma**2/(2.*ttail))/ttail) * spc.erfc((x-mu+sigma**2/ttail)/(np.sqrt(2.)*sigma))
    return gaus + tail

def gausTailConst(x, ngaus, mu, sigma, ntail, ttail, p0):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    tail = (ntail/(2.*ttail)) * np.exp((x-mu+sigma**2/(2.*ttail))/ttail) * spc.erfc((x-mu+sigma**2/ttail)/(np.sqrt(2.)*sigma))
    return gaus + tail + p0

def gausTailLin(x, ngaus, mu, sigma, ntail, ttail, p0, p1):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    tail = (ntail/(2.*ttail)) * np.exp((x-mu+sigma**2/(2.*ttail))/ttail) * spc.erfc((x-mu+sigma**2/ttail)/(np.sqrt(2.)*sigma))
    return gaus + tail + p0 + p1*x

def gausTailQuad(x, ngaus, mu, sigma, ntail, ttail, p0, p1, p2):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    tail = (ntail/(2.*ttail)) * np.exp((x-mu+sigma**2/(2.*ttail))/ttail) * spc.erfc((x-mu+sigma**2/ttail)/(np.sqrt(2.)*sigma))
    return gaus + tail + p0 + p1*x + p2*x**2

def gausStep(x, ngaus, mu, sigma, cstep):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    step = cstep * spc.erfc((x-mu)/(np.sqrt(2.)*sigma))
    return gaus + step

def gausStepConst(x, ngaus, mu, sigma, cstep, p0):
    #gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    step = cstep * spc.erfc((x-mu)/(np.sqrt(2.)*sigma))
    return gaus(x, ngaus, mu, sigma) + step + p0

def gausStepLin(x, ngaus, mu, sigma, cstep, p0, p1):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    step = cstep * spc.erfc((x-mu)/(np.sqrt(2.)*sigma))
    return gaus + step + p0 + p1*x

def gausStepQuad(x, ngaus, mu, sigma, cstep, p0, p1, p2):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    step = cstep * spc.erfc((x-mu)/(np.sqrt(2.)*sigma))
    return gaus + step + p0 + p1*x + p2*x**2

def gausHyper(x, ngaus, mu, sigma, ntail, ttail, cstep):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    tail = (ntail/(2.*ttail)) * np.exp((x-mu+sigma**2/(2.*ttail))/ttail) * spc.erfc((x-mu+sigma**2/ttail)/(np.sqrt(2.)*sigma))
    step = cstep * spc.erfc((x-mu)/(np.sqrt(2.)*sigma))
    return gaus + tail + step

def gausHyperConst(x, ngaus, mu, sigma, ntail, ttail, cstep, p0):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    tail = (ntail/(2.*ttail)) * np.exp((x-mu+sigma**2/(2.*ttail))/ttail) * spc.erfc((x-mu+sigma**2/ttail)/(np.sqrt(2.)*sigma))
    step = cstep * spc.erfc((x-mu)/(np.sqrt(2.)*sigma))
    return gaus + tail + step + p0

def gausHyperLin(x, ngaus, mu, sigma, ntail, ttail, cstep, p0, p1):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    tail = (ntail/(2.*ttail)) * np.exp((x-mu+sigma**2/(2.*ttail))/ttail) * spc.erfc((x-mu+sigma**2/ttail)/(np.sqrt(2.)*sigma))
    step = cstep * spc.erfc((x-mu)/(np.sqrt(2.)*sigma))
    return gaus + tail + step + p0 + p1*x

def gausHyperQuad(x, ngaus, mu, sigma, ntail, ttail, cstep, p0, p1, p2):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    tail = (ntail/(2.*ttail)) * np.exp((x-mu+sigma**2/(2.*ttail))/ttail) * spc.erfc((x-mu+sigma**2/ttail)/(np.sqrt(2.)*sigma))
    step = cstep * spc.erfc((x-mu)/(np.sqrt(2.)*sigma))
    return gaus + tail + step + p0 + p1*x + p2*x**2

peakfnc = {'gaus':  {'no'    : gaus,
                     'const' : gausConst,
                     'lin'   : gausLin,
                     'quad'  : gausQuad},
           'tail':  {'no'    : gausTail,
                     'const' : gausTailConst,
                     'lin'   : gausTailLin,
                     'quad'  : gausTailQuad},
           'step':  {'no'    : gausStep,
                     'const' : gausStepConst,
                     'lin'   : gausStepLin,
                     'quad'  : gausStepQuad},
           'hyper': {'no'    : gausHyper,
                     'const' : gausHyperConst,
                     'lin'   : gausHyperLin,
                     'quad'  : gausHyperQuad}}


#A/E FUNCTIONS

def gausDoubleTail(x, ngaus, mu, sigma, ntail, ttail, ntail2, ttail2):
    gaus = (ngaus/(np.sqrt(2.*cnt.pi)*sigma)) * np.exp(-(x-mu)**2/(2.*sigma**2))
    tail1 = (ntail/(2.*ttail)) * np.exp((x-mu+sigma**2/(2.*ttail))/ttail) * spc.erfc((x-mu+sigma**2/ttail)/(np.sqrt(2.)*sigma))
    tail2 = (ntail2/(2.*ttail2)) * np.exp((x-mu+sigma**2/(2.*ttail2))/ttail2) * spc.erfc((x-mu+sigma**2/ttail2)/(np.sqrt(2.)*sigma))
    return gaus + tail1 + tail2


#ODR FUNCTIONS f(p,x)

def odrconst(p, x):
    return p[0] + 0*x

def odrlin(p, x):
    return p[0] + p[1]*x

def odrquad(p, x):
    return p[0] + p[1]*x + p[2]*x**2 

def odrcube(p, x):
    return p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3

def odrquart(p, x):
    return p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4

polyorder = {'lin'  : (odrlin, 2),
             'quad' : (odrquad, 3),
             'cube' : (odrcube, 4),
             'quart': (odrquart, 5)}