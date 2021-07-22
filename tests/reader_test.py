import numpy as np
import os
import math
from begepro.rw import CAENhandler
from begepro.dspro import filters as flt

'''
base_corr_wf = raw_wf - np.mean(raw_wf[0:100])
pz_corr_wf = flt.pz_corr(base_corr_wf)
shaped_wf = flt.trap_filter(pz_corr_wf, 500, 250)
'''

def main():

    measureName = '228Th-grafico-tesi-im260421_1'
    measurePath = '/media/alessandro/Volume/BEGe-DPP/test2/' + measureName + '/FILTERED/DataF_CH1@DT5725SB_10806_' + measureName
    
    counter = 0

    ph_list = list()
    e_list = list()
    a_list = list()
    ae_list = list()

    for i in range(265):

        if i == 0: rd = CAENhandler.compassReader(measurePath + '.bin', calibrated=True)
        else: rd = CAENhandler.compassReader(measurePath + '_' + str(i) + '.bin', calibrated=True)

        print(measureName + '_' + str(i))
        
        while True:

            data = rd.get()
            if data is None: break

            raw_wf = np.array(data['trace'])
            curr = flt.curr_filter(raw_wf)
                
            ph = data['pulseheight']
            e = data['energy']
            a = np.max(curr)
            ae = a / ph
                
            ph_list.append(ph)
            e_list.append(e)
            a_list.append(a)
            ae_list.append(ae)
            
            #e_list.append(np.max(raw_wf))

            counter += 1
            if counter%10000 == 0: print('{sgn} signals processed...'.format(sgn=counter))

    np.save('psd/' + measureName, np.transpose(np.array([ph_list, e_list, a_list, ae_list])))
    #np.save('muoni/' + measure, np.array(e_list))

    return


if __name__ == '__main__':
    main()