#!/usr/bin/env python

#Duncan Campbell
#Yale University
#August 8,2014
#calculate projected cross correlation function for sdss groups x cfhtlens photometric 
#  galaxies

from __future__ import print_function, division
import sys
import numpy as np
import custom_utilities as cu
from HAS.TPCF.proj_cross_TPCF_serial import proj_cross_npairs_serial 
import h5py
from astropy.cosmology import FlatLambdaCDM

def main():
    
    if len(sys.argv)>1:
        field = sys.argv[1]
    else: field = 'W3'
    if len(sys.argv)>2:
        catalogue = sys.argv[2]
    else: catalogue = 'sample3_L_model'

    #define cosmology
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
 
    #import cfhtlens catalogues
    filepath = '/scratch/dac29/output/processed_data/CFHTLens/'
    f = h5py.File(filepath+'catalogues/'+field+'.hdf5', 'r')
    W = f.get(field)
    R = np.load(filepath+'random_catalogues/'+field+'_randoms'+'.npy')
    #print(W.dtype.names)
    #print(R.dtype.names)

    #import sdss group catalogue
    filepath = '/scratch/dac29/output/processed_data/yang_groupcat/'
    f =  h5py.File(filepath+'catalogues/'+catalogue+'.hdf5', 'r')
    GC = f.get(catalogue)
    #print(GC.dtype.names)

    #choose group sample to cross correlate with
    #choose galaxies in the field
    filepath = '/scratch/dac29/output/processed_data/CFHTLens/'
    field_boundary = np.load(filepath+'field_boundaries/'+field +'.npy')
    condition_1 = cu.inside(GC['RAgal'],GC['DECgal'],field_boundary)
    #choose central galaxies
    condition_2 = (GC['Rproj_L'] == 0)
    #cut off low redshift(for now)
    condition_3 = (GC['GROUP_Z'] >= 0.02)
    #combine conditionals
    condition = (condition_1 & condition_2 & condition_3)
    GC = GC[condition]
    
    #choose cfhtlens sample
    
    #choose random sample

    print(np.min(GC['GROUP_Z']), np.max(GC['GROUP_Z']))

    data_1 = np.column_stack((GC['RAgal'],GC['DECgal'],GC['GROUP_Z']))
    data_2 = np.column_stack((W['ALPHA_J2000'],W['DELTA_J2000']))
    r_bins = np.logspace(-2,1,10)
    print(r_bins)
    result = proj_cross_npairs_serial(data_1, data_2, r_bins, cosmo,\
                             weights_1=None, weights_2=None,\
                             wf=None, aux_1=None, aux_2=None)

    print(result)



if __name__ == '__main__':
    main()
