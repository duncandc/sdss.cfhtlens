#!/usr/bin/env python

#Duncan Campbell
#Yale University
#August 8,2014
#calculate the projected cross correlation function for sdss groups x cfhtlens galaxies 

from __future__ import print_function, division
import sys
import numpy as np
import custom_utilities as cu
from halotools.mock_observables.observables import projected_cross_two_point_correlation_function
import h5py
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
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
    fr = 100
    
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
    condition_1 = ((W['MAG_y']<24.5) & (W['MAG_y']!=-99))
    condition_2 = ((W['MAG_i']<24.5) & (W['MAG_i']!=-99))
    condition_12 = (condition_1 | condition_2)
    condition_3 = (W['Z_B']<0.2)
    condition_4 = (W['MASK']==0)
    condition = ((condition_12 & condition_3) & condition_4)
    W = W[condition]
    if DS==True:
        N = len(W)
        np.random.seed(0)
        inds = np.random.permutation(np.arange(0,N))
        inds = inds[0:N//fw]
        W = W[inds]
    
    if rank==0: print("N2: {0}".format(len(W)))
    
    
    fs = 1000
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
    
    """ 
    if rank==0:
        #plot the field and randoms as a sanity check
        fig = plt.figure(figsize=plt.figaspect(1))
        #fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(2, 2, 1)
        #ax.plot(R['ra'],R['dec'],'.',color='black', ms=2)
        ax.plot(GC['RAgal'],GC['DECgal'],'.',color='blue', ms=2)
        ax.plot(W['ALPHA_J2000'],W['DELTA_J2000'],'.',color='red', ms=2, alpha=0.25)
        ax.set_xlabel('ra')
        ax.set_ylabel('dec')
        ax.set_xlim([206,222])
        ax.set_ylim([50,59])
        #ax.legend(('randoms','sdss centrals','cfhtlens'))
        ax = fig.add_subplot(2, 2, 2)
        ax.plot(W['ALPHA_J2000'],W['DELTA_J2000'],'.',color='red', ms=2, alpha=0.25)
        ax.set_xlabel('ra')
        ax.set_ylabel('dec')
        ax.set_xlim([206,222])
        ax.set_ylim([50,59])
        ax = fig.add_subplot(2, 2, 3)
        ax.plot(R['ra'],R['dec'],'.',color='black', ms=2, alpha=0.25)
        ax.set_xlabel('ra')
        ax.set_ylabel('dec')
        ax.set_xlim([206,222])
        ax.set_ylim([50,59])
        #place on surface of a unit sphere to plot for sanity check
        from halotools.utils.spherical_geometry import spherical_to_cartesian, chord_to_cartesian
        from mpl_toolkits.mplot3d import Axes3D
        xyz_1 = np.empty((len(data_1),3))
        xyz_1[:,0],xyz_1[:,1],xyz_1[:,2] = spherical_to_cartesian(data_1[:,0], data_1[:,1])
        xyz_2 = np.empty((len(data_2),3))
        xyz_2[:,0],xyz_2[:,1],xyz_2[:,2] = spherical_to_cartesian(data_2[:,0], data_2[:,1])
        xyz_randoms = np.empty((len(randoms),3))
        xyz_randoms[:,0],xyz_randoms[:,1],xyz_randoms[:,2] = spherical_to_cartesian(randoms[:,0], randoms[:,1])
    
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        #plot a spherical surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 1.0 * np.outer(np.cos(u), np.sin(v))
        y = 1.0 * np.outer(np.sin(u), np.sin(v))
        z = 1.0 * np.outer(np.ones(np.size(u)), np.cos(v))
        #plot points on surface
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='grey',alpha=0.2)
        ax.plot(xyz_randoms[:,0],xyz_randoms[:,1],xyz_randoms[:,2],'.',color='black',ms=2)
        ax.plot(xyz_1[:,0],xyz_1[:,1],xyz_1[:,2],'.',color='blue',ms=2)
        ax.plot(xyz_2[:,0],xyz_2[:,1],xyz_2[:,2],'.',color='red',ms=2)
        #ax.set_xlim([min(xyz_1[:,0]),max(xyz_1[:,0])])
        #ax.set_ylim([min(xyz_1[:,1]),max(xyz_1[:,1])])
        #ax.set_zlim([min(xyz_1[:,2]),max(xyz_1[:,2])])
        plt.show(block=False)
    """
    
    #define angular bins
    r_bins = np.logspace(-2,0,25)
    bin_centers = (r_bins[:-1]+r_bins[1:])/2.0
    if rank==0:
        print("radial bins:",r_bins)
    
    result = projected_cross_two_point_correlation_function(data_1, z_gal, data_2, r_bins, cosmo=cosmo, N_theta_bins=5, randoms=randoms,\
                                                            N_threads=1, estimator='Davis-Peebles',\
                                                            comm=comm, max_sample_size=int(1e8))
                                                
    
    if rank==0:
        print(result)
        print(bin_centers)
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