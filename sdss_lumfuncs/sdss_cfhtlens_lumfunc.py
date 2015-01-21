#!/usr/bin/env python

#Duncan Campbell
#Yale University
#August 8,2014
#calculate the projected cross correlation function for sdss groups x cfhtlens galaxies 

from __future__ import print_function, division
import sys
import h5py
import numpy as np
import custom_utilities as cu
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from cross_tpcf.projected_cross_tpcf import projected_cross_two_point_correlation_function
from mpi4py import MPI


def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if len(sys.argv)>1:
        field = sys.argv[1]
    else: field = 'W3'
    if len(sys.argv)>2:
        catalogue = sys.argv[2]
    else: catalogue = 'sample3_L_model'

    savepath = cu.get_plot_path()+'/analysis/sdss.cfhtlens/cross_correlations/'

    #define cosmology
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    
    #down sample the number of points? Use this for testing purposes...
    DS = True
    fw = 10
    fr = 1000
    
    #import cfhtlens catalogues
    filepath = cu.get_output_path()+'processed_data/CFHTLens/'
    f = h5py.File(filepath+'catalogues/'+field+'_abbreviated.hdf5', 'r')
    W = f.get(field)
    filepath = cu.get_output_path()+'processed_data/CFHTLens/'
    R = np.load(filepath+'random_catalogues/'+field+'_randoms'+'.npy')
    print(W.dtype.names)
    #print(R.dtype.names)
    
    #import sdss group catalogue
    filepath = cu.get_output_path()+'processed_data/yang_groupcat/'
    f =  h5py.File(filepath+'catalogues/'+catalogue+'.hdf5', 'r')
    GC = f.get(catalogue)
    #print(GC.dtype.names)
    
    #choose group sample to cross correlate with
    #choose galaxies in the field
    filepath = cu.get_output_path()+'processed_data/CFHTLens/'
    field_boundary = np.load(filepath+'field_boundaries/'+field +'.npy')
    condition_1 = cu.inside(GC['RAgal'],GC['DECgal'],field_boundary)
    #choose central galaxies
    condition_2 = (GC['Rproj_L'] == 0)
    #cut off low redshift(for now)
    condition_3 = (GC['GROUP_Z'] >= 0.02)
    #combine conditionals
    condition = (condition_1 & condition_2 & condition_3)
    GC = GC[condition]
    
    if rank==0: print("N1: {0}".format(len(GC)))
    
    #choose cfhtlens sample
    condition_1 = ((W['MAG_y']<24.5) & (W['MAG_y']>0))
    condition_2 = ((W['MAG_i']<24.5) & (W['MAG_i']>0))
    condition_12 = (condition_1 | condition_2)
    W = W[condition_12]
    condition_3 = (W['Z_B']<0.2)
    condition_4 = (W['MASK']==0)
    condition = (condition_3 & condition_4)
    W = W[condition]
    if DS==True:
        N = len(W)
        np.random.seed(0)
        inds = np.random.permutation(np.arange(0,N))
        inds = inds[0:N//fw]
        W = W[inds]
    
    if rank==0: print("N2: {0}".format(len(W)))
    
    #choose random sample
    condition_1 = (R['flag']==0)
    condition = condition_1
    R = R[condition]
    if DS==True:
        N = len(R)
        np.random.seed(0)
        inds = np.random.permutation(np.arange(0,N))
        inds = inds[0:N//fr]
        R = R[inds]
        
    if rank==0: print("Nran: {0}".format(len(R)))
    
    #get the data into the appropriate form
    data_1 = np.column_stack((GC['RAgal'],GC['DECgal']))
    z_gal = GC['ZGAL']
    data_2 = np.column_stack((W['ALPHA_J2000'],W['DELTA_J2000']))
    randoms = np.column_stack((R['ra'],R['dec']))
    #define weighting function
    from halotools.mock_observables.spatial.kdtrees.ckdtree import Function
    class MyFunction(Function):
        def evaluate(self, x, y, a, b):
            if b<a: return x*y
            else: return 0.0 
    """
    class MyFunction(Function):
        def evaluate(self, x, y, a, b):
            return x*y
    """
    
    
    #define weights
    #weights1 = np.random.random_sample(len(GC))
    weights1 = np.ones(len(GC))
    #weights2 = np.random.random_sample(len(W))
    weights2 = np.ones(len(W))
    #weights_randoms = np.random.random_sample(len(randoms))
    weights_randoms = np.ones(len(randoms))
    #maximum apparent magnitude to make cut at redshift of galaxy
    dL = cosmo.luminosity_distance(GC['ZGAL']).value
    Mag_threshold = -17.0
    aux1 = cu.absolute_to_apparent_magnitude(Mag_threshold,dL)
    #apparent magnitude of cfhtlens galaxy
    aux2 = np.maximum(W['MAG_i'],W['MAG_y'])
    inds = np.random.random_integers(0,len(W)-1,len(randoms))
    aux_randoms = np.maximum(W['MAG_i'][inds],W['MAG_y'][inds])
    
    print(np.min(aux1), np.max(aux1))
    print(np.min(aux2), np.max(aux2))
    print(np.min(aux_randoms), np.max(aux_randoms))
    
    print(MyFunction().evaluate(1,2,3,4))
    
    sys.exit()
        
    #define angular bins
    r_bins = np.logspace(-2,0,25)
    bin_centers = (r_bins[:-1]+r_bins[1:])/2.0
    
    result = projected_cross_two_point_correlation_function(data_1, z_gal, data_2, r_bins,\
                                                            cosmo=cosmo, N_theta_bins=5,\
                                                            randoms=randoms, N_threads=1,\
                                                            weights1=weights1, weights2=weights2,\
                                                            weights_randoms=weights_randoms,\
                                                            wf=MyFunction(),\
                                                            aux1=aux1, aux2=aux2,\
                                                            estimator='Davis-Peebles',\
                                                            comm=comm, max_sample_size=int(1e8))
    
    print(result)
    
     
    if rank==0:
        fig1 = plt.figure()
        plt.plot(bin_centers,result,'o-')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$r_{\rm proj} [Mpc]$')
        plt.ylabel(r'$\omega(r_{\rm proj})$')
        filename = 'projected_correlation_'+field+'.pdf'
        plt.savefig(savepath+filename)
        plt.show()
    
                                                            
if __name__ == '__main__':
    main()