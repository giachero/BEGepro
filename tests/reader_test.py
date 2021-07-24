import numpy as np
import os
import sys
import math
import argparse
from begepro.rw import CAENhandler
from begepro.dspro import filters as flt

'''
base_corr_wf = raw_wf - np.mean(raw_wf[0:100])
pz_corr_wf = flt.pz_corr(base_corr_wf)
shaped_wf = flt.trap_filter(pz_corr_wf, 500, 250)
'''

def main():

    usage='./reader_test.py --loc /path/where/the/measurement/directory/is/ --meas measurementName --nfiles numberOfFiles'
    parser = argparse.ArgumentParser(description='Test script to read and analyze BEGe signals from CAEN desktop digitizer', usage=usage)

    # Check for arguments
    #if len(sys.argv[1:]) == 0:
    #    parser.error('No argument given!')
    #    return

    parser.add_argument("-l", "--loc",    dest="dirloc",   type=str, help="Location of measurement directory", required = True)
    parser.add_argument("-m", "--meas",   dest="measname", type=str, help="Measurement name",                  required = True)
    parser.add_argument("-n", "--nfiles", dest="nfiles",   type=int, help="Number of files to analyze",        required = True)
    parser.add_argument("-d", "--dir",    dest="savedir",  type=str, help="Path where to save analysis",       default  = '')
    
    args = parser.parse_args()

    path = args.dirloc + args.measname + '/FILTERED/DataF_CH1@DT5725SB_10806_' + args.measname

    ph_list = list()
    e_list = list()
    a_list = list()
    ae_list = list()

    counter = 0

    print('+++ Start of analysis +++')

    for i in range(args.nfiles):

        if i == 0: rd = CAENhandler.compassReader(path + '.bin',                calibrated=True)
        else:      rd = CAENhandler.compassReader(path + '_' + str(i) + '.bin', calibrated=True)

        print('*** Start of file ' + str(i+1) + '/' + args.nfiles + ' ***')
        
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
            
            counter += 1
            if counter%10000 == 0: print('{sgn} signals processed...'.format(sgn=counter))

        print('*** End of file ' + str(i+1) + '/' + args.nfiles + ' ***')

    np.save(args.savedir + args.measname, np.transpose(np.array([ph_list, e_list, a_list, ae_list])))

    print('+++ End of analysis +++')

    return


if __name__ == '__main__':
    main()