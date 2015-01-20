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

from mpi4py import MPI
####import modules########################################################################
from math import pi, gamma
from multiprocessing import Pool
from halotools.mock_observables.pair_counters.npairs_mpi import npairs, wnpairs, specific_wnpairs, jnpairs
##########################################################################################

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
    
    result = projected_cross_two_point_correlation_function(data_1, z_gal, data_2, r_bins,\ 
                                                            cosmo=cosmo, N_theta_bins=5,\
                                                            randoms=randoms, N_threads=1,\
                                                            estimator='Davis-Peebles',\
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


def projected_cross_two_point_correlation_function(sample1, z, sample2, r_bins, cosmo=None, 
                                                   N_theta_bins=10, randoms=None,
                                                   weights1=None, weights2=None, weights_randoms=None, 
                                                   max_sample_size=int(1e6),
                                                   estimator='Natural',
                                                   N_threads=1, comm=None):
    """ Calculate the projected cross two point correlation function between a spec-z set 
    and a photometric set 
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x 2 numpy array containing ra,dec positions of Npts. 
    
    theta_bins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(theta_bins) = N_theta_bins + 1.
    
    sample2 : array_like, optional
        Npts x 2 numpy array containing ra,dec positions of Npts.
    
    randoms : array_like, optional
        Nran x 2 numpy array containing ra,dec positions of Npts.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled 
        such that the subsamples are (roughly) equal to max_sample_size. 
        Subsamples will be passed to the pair counter in a simple loop, 
        and the correlation function will be estimated from the median pair counts in each bin.
    
    estimator: string, optional
        options: 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
    
    N_thread: int, optional
        number of threads to use in calculation.

    comm: mpi Intracommunicator object, optional
    
    do_auto: boolean, optional
        do auto-correlation?
    
    do_cross: boolean, optional
        do cross-correlation?
    
    Returns 
    -------
    angular correlation_function : array_like
        array containing correlation function :math:`\\xi` computed in each of the Nrbins 
        defined by input `rbins`.

        :math:`1 + \\xi(r) \equiv DD / RR`, 
        where `DD` is calculated by the pair counter, and RR is counted by the internally 
        defined `randoms` if no randoms are passed as an argument.

        If sample2 is passed as input, three arrays of length Nrbins are returned: two for
        each of the auto-correlation functions, and one for the cross-correlation function. 

    """
    #####notes#####
    #The pair counter returns all pairs, including self pairs and double counted pairs 
    #with separations less than r. If PBCs are set to none, then period=np.inf. This makes
    #all distance calculations equivalent to the non-periodic case, while using the same 
    #periodic distance functions within the pair counter.
    ###############
    
    do_auto=False
    do_cross=True
    
    if comm!=None:
        rank=comm.rank
    else: rank=0
    if N_threads>1:
        pool = Pool(N_threads)
    
    def list_estimators(): #I would like to make this accessible from the outside. Know how?
        estimators = ['Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay']
        return estimators
    estimators = list_estimators()
    
    #process input parameters
    sample1 = np.asarray(sample1)
    if np.all(sample2 != None): sample2 = np.asarray(sample2)
    else: sample2 = sample1
    if np.all(randoms != None): 
        randoms = np.asarray(randoms)
        PBCs = False
    else: PBCs = True #assume full sky coverage
    r_bins = np.asarray(r_bins)
        
    #down sample is sample size exceeds max_sample_size.
    if (len(sample2)>max_sample_size) & (not np.all(sample1==sample2)):
        inds = np.arange(0,len(sample2))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample2 = sample2[inds]
        print('down sampling sample2...')
    if len(sample1)>max_sample_size:
        inds = np.arange(0,len(sample1))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample1 = sample1[inds]
        print('down sampling sample1...')
    
    if np.shape(r_bins) == ():
        theta_bins = np.array([r_bins])
    
    k = 2 #only 2-dimensions: ra,dec
    if np.shape(sample1)[-1] != k:
        raise ValueError('angular correlation function requires 2-dimensional data')
    
    #check for input parameter consistency
    if np.all(sample2 != None) & (sample1.shape[-1]!=sample2.shape[-1]):
        raise ValueError('Sample 1 and sample 2 must have same dimension.')
    if estimator not in estimators: 
        raise ValueError('Must specify a supported estimator. Supported estimators are:{0}'
        .value(estimators))

    #If PBCs are defined, calculate the randoms analytically. Else, the user must specify 
    #randoms and the pair counts are calculated the old fashion way.
    def random_counts(sample1, sample2, randoms, theta_bins, PBCs, N_threads, do_RR, do_DR, comm):
        """
        Count random pairs.
        """
        def cap_area(C):
            """
            Calculate angular area of a spherical cap with chord length c
            """
            theta = 2.0*np.arcsin(C/2.0)
            return 2.0*np.pi*(1.0-np.cos(theta))
        
        #No PBCs, randoms must have been provided.
        if PBCs==False:
            if comm!=None:
                if do_RR==True:
                    if rank==0: print('Running MPI pair counter for RR with {0} processes.'.format(comm.size))
                    RR = specific_wnpairs(randoms, randoms, theta_bins, comm=comm)
                    RR = np.diff(RR)
                else: RR=None
                if do_DR==True:
                    if rank==0: print('Running MPI pair counter for D1R with {0} processes.'.format(comm.size))
                    D1R = specific_wnpairs(sample1, randoms, theta_bins, comm=comm)
                    D1R = np.diff(D1R)
                else: D1R=None
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    print('manually skipping D2R right now.')
                    if True==False:
                    #if do_DR==True:
                        if rank==0: print('Running MPI pair counter for D2R with {0} processes.'.format(comm.size))
                        D2R = specific_wnpairs(sample2, randoms, theta_bins, comm=comm)
                        D2R = np.diff(D2R)
                    else: D2R=None
            elif N_threads==1:
                if do_RR==True:
                    RR = specific_wnpairs(randoms, randoms, theta_bins)
                    RR = np.diff(RR)
                else: RR=None
                if do_DR==True:
                    D1R = specific_wnpairs(sample1, randoms, theta_bins)
                    D1R = np.diff(D1R)
                else: D1R=None
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    if do_DR==True:
                        D2R = specific_wnpairs(sample2, randoms, theta_bins)
                        D2R = np.diff(D2R)
                    else: D2R=None
            else:
                if do_RR==True:
                    args = [[chunk,randoms,theta_bins] for chunk in np.array_split(randoms,N_threads)]
                    RR = np.sum(pool.map(_specific_wnpairs_wrapper,args),axis=0)
                    RR = np.diff(RR)
                else: RR=None
                if do_DR==True:
                    args = [[chunk,randoms,theta_bins] for chunk in np.array_split(sample1,N_threads)]
                    D1R = np.sum(pool.map(_specific_wnpairs_wrapper,args),axis=0)
                    D1R = np.diff(D1R)
                else: D1R=None
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    if do_DR==True:
                        args = [[chunk,randoms,theta_bins] for chunk in np.array_split(sample2,N_threads)]
                        D2R = np.sum(pool.map(_specific_wnpairs_wrapper,args),axis=0)
                        D2R = np.diff(D2R)
                    else: D2R=None
            
            return D1R, D2R, RR
        #PBCs and no randoms--calculate randoms analytically.
        elif PBCs==True:
            #do volume calculations
            dv = cap_area(theta_bins) #volume of spheres
            dv = np.diff(dv) #volume of shells
            global_area = 4.0*np.pi
            
            #calculate randoms for sample1
            N1 = np.shape(sample1)[0]
            rho1 = N1/global_area
            D1R = (N1)*(dv*rho1) #read note about pair counter
            
            #if not calculating cross-correlation, set RR exactly equal to D1R.
            if np.all(sample1 == sample2):
                D2R = None
                RR = D1R #in the analytic case, for the auto-correlation, DR==RR.
            else: #if there is a sample2, calculate randoms for it.
                N2 = np.shape(sample2)[0]
                rho2 = N2/global_area
                D2R = N2*(dv*rho2) #read note about pair counter
                #calculate the random-random pairs.
                NR = N1*N2
                rhor = NR/global_area
                RR = (dv*rhor) #RR is only the RR for the cross-correlation.

            return D1R, D2R, RR
        else:
            raise ValueError('Un-supported combination of PBCs and randoms provided.')
    
    def pair_counts(sample1, sample2, weights1, weights2, theta_bins, N_threads, do_auto, do_cross, do_DD, comm):
        """
        Count data pairs: D1D1, D1D2, D2D2.  If a comm object is passed, the code uses a
        MPI pair counter.  Else if N_threads==1, the calculation is done serially.  Else,
        the calculation is done on N_threads threads. 
        """
        if comm!=None:
            if do_auto==True:
                if rank==0: print('Running MPI pair counter for D1D1 with {0} processes.'.format(comm.size))
                D1D1 = specific_wnpairs(sample1, sample1, theta_bins, period=None, weights1=weights1, weights2=weights2, wf=None, comm=comm)
                D1D1 = np.diff(D1D1)
            else: D1D1=None
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                if do_cross==True:
                    if rank==0: print('Running MPI pair counter for D1D2 with {0} processes.'.format(comm.size))
                    D1D2 = specific_wnpairs(sample1, sample2, theta_bins, period=None, weights1=weights1, weights2=weights2, wf=None, comm=comm)
                    D1D2 = np.diff(D1D2)
                else: D1D2=None
                if do_auto==True:
                    if rank==0: print('Running MPI pair counter for D2D2 with {0} processes.'.format(comm.size))
                    D2D2 = specific_wnpairs(sample2, sample2, theta_bins, period=None, weights1=weights2, weights2=weights2, wf=None, comm=comm)
                    D2D2 = np.diff(D2D2)
                else: D2D2=False
        elif N_threads==1:
            if do_auto==True:
                D1D1 = specific_wnpairs(sample1, sample1, theta_bins, period=None, weights1=weights1, weights2=weights1, wf=None)
                D1D1 = np.diff(D1D1)
            else: D1D1=None
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                if do_cross==True:
                    D1D2 = specific_wnpairs(sample1, sample2, theta_bins, period=None, weights1=weights1, weights2=weights2, wf=None)
                    D1D2 = np.diff(D1D2)
                else: D1D2=None
                if do_auto==True:
                    D2D2 = specific_wnpairs(sample2, sample2, theta_bins, period=None, weights1=weights2, weights2=weights2, wf=None)
                    D2D2 = np.diff(D2D2)
                else: D2D2=False
        else:
            inds1 = np.arange(0,len(sample1)) #indices into sample1
            inds2 = np.arange(0,len(sample2)) #indices into sample2
            if do_auto==True:
                #split sample1 into subsamples for list of args to pass to the pair counter
                args = [[sample1[chunk],sample1,theta_bins, None, weights1[chunk], weights1, None] for chunk in np.array_split(inds1,N_threads)]
                D1D1 = np.sum(pool.map(_specific_wnpairs_wrapper,args),axis=0)
                D1D1 = np.diff(D1D1)
            else: D1D1=None
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                if do_cross==True:
                    #split sample1 into subsamples for list of args to pass to the pair counter
                    args = [[sample1[chunk],sample2,theta_bins, None, weights1[chunk], weights2, None] for chunk in np.array_split(inds1,N_threads)]
                    D1D2 = np.sum(pool.map(_specific_wnpairs_wrapper,args),axis=0)
                    D1D2 = np.diff(D1D2)
                else: D1D2=None
                if do_auto==True:
                   #split sample2 into subsamples for list of args to pass to the pair counter
                    args = [[sample2[chunk],sample2,theta_bins, None, weights2[chunk], weights2, None] for chunk in np.array_split(inds2,N_threads)]
                    D2D2 = np.sum(pool.map(_specific_wnpairs_wrapper,args),axis=0)
                    D2D2 = np.diff(D2D2)
                else: D2D2=None

        return D1D1, D1D2, D2D2
        
    def TP_estimator(DD,DR,RR,ND1,ND2,NR1,NR2,estimator):
        """
        two point correlation function estimator
        """
        if estimator == 'Natural': #DD/RR-1
            factor = ND1*ND2/(NR1*NR2)
            xi = (1.0/factor)*DD/RR - 1.0
        elif estimator == 'Davis-Peebles': #DD/DR-1
            factor = ND1*ND2/(ND1*NR2)
            xi = (1.0/factor)*DD/DR - 1.0
        elif estimator == 'Hewett': #(DD-DR)/RR
            factor1 = ND1*ND2/(NR1*NR2)
            factor2 = ND1*NR2/(NR1*NR2)
            xi = (1.0/factor1)*DD/RR - (1.0/factor2)*DR/RR 
        elif estimator == 'Hamilton': #DDRR/DRDR-1
            xi = (DD*RR)/(DR*DR) - 1.0
        elif estimator == 'Landy-Szalay': #(DD - 2.0*DR + RR)/RR
            factor1 = ND1*ND2/(NR1*NR2)
            factor2 = ND1*NR2/(NR1*NR2)
            xi = (1.0/factor1)*DD/RR - (1.0/factor2)*2.0*DR/RR + 1.0
        else: 
            raise ValueError("unsupported estimator!")
        return xi
    
    def TP_estimator_requirements(estimator):
        """
        return booleans indicating which pairs need to be counted for the chosen estimator
        """
        if estimator == 'Natural':
            do_DD = True
            do_DR = False
            do_RR = True
        elif estimator == 'Davis-Peebles':
            do_DD = True
            do_DR = True
            do_RR = False
        elif estimator == 'Hewett':
            do_DD = True
            do_DR = True
            do_RR = True
        elif estimator == 'Hamilton':
            do_DD = True
            do_DR = True
            do_RR = True
        elif estimator == 'Landy-Szalay':
            do_DD = True
            do_DR = True
            do_RR = True
        else: 
            raise ValueError("unsupported estimator!")
        return do_DD, do_DR, do_RR
              
    if np.all(randoms != None):
        N1 = len(sample1)
        N2 = len(sample2)
        NR = len(randoms)
    else: 
        N1 = 1.0
        N2 = 1.0
        NR = 1.0
    
    def proj_r_to_angular_bins(r_bins, z, N_sample, cosmo):
        """
        define angular bins given r_proj bins and redshift range.
        parameters
            r_bins: np.array, projected radial bins in Mpc
            N_sample: int, oversample rate of theta bins
            cosmo: astropy cosmology object defining cosmology
        returns:
            theta_bins: np.array, angular bins in radians
        """
    
        N_r_bins = len(r_bins)
        N_theta_bins = N_sample * N_r_bins
    
        #find maximum theta
        X_min = cosmo.comoving_distance(np.min(z)).value
        max_theta = np.max(r_bins)/(X_min/(1.0+np.min(z)))
    
        #find minimum theta
        X_max = cosmo.comoving_distance(np.max(z)).value
        min_theta = np.min(r_bins)/(X_max/(1.0+np.max(z)))
    
        theta_bins = np.linspace(np.log10(min_theta), np.log10(max_theta), N_theta_bins)
        theta_bins = 10.0**theta_bins
    
        return theta_bins*180.0/np.pi
    
    do_DD, do_DR, do_RR = TP_estimator_requirements(estimator)
    
    theta_bins = proj_r_to_angular_bins(r_bins, z, N_theta_bins, cosmo)
    if rank==0:
        print("bins")
        print(r_bins)
        print(theta_bins)
    
    #convert angular coordinates into cartesian coordinates
    from halotools.utils.spherical_geometry import spherical_to_cartesian, chord_to_cartesian
    xyz_1 = np.empty((len(sample1),3))
    xyz_2 = np.empty((len(sample2),3))
    xyz_1[:,0],xyz_1[:,1],xyz_1[:,2] = spherical_to_cartesian(sample1[:,0], sample1[:,1])
    xyz_2[:,0],xyz_2[:,1],xyz_2[:,2] = spherical_to_cartesian(sample2[:,0], sample2[:,1])
    if PBCs==False:
        xyz_randoms = np.empty((len(randoms),3))
        xyz_randoms[:,0],xyz_randoms[:,1],xyz_randoms[:,2] = spherical_to_cartesian(randoms[:,0], randoms[:,1])
    else: xyz_randoms=None
    
    #convert angular bins to cartesian distances
    c_bins = chord_to_cartesian(theta_bins, radians=False)
    
    #count pairs
    if rank==0: print('counting data pairs...')
    D1D1,D1D2,D2D2 = pair_counts(xyz_1, xyz_2, weights1, weights2, c_bins, N_threads, do_auto, do_cross, do_DD, comm)
    if rank==0: print('counting random pairs...')
    D1R, D2R, RR = random_counts(xyz_1, xyz_2, xyz_randoms, c_bins, PBCs, N_threads, do_RR, do_DR, comm)
    if rank==0: print('done counting pairs.')
    
    #covert angular pair counts to projected pair counts
    #comoving distance to sample 1
    X = cosmo.comoving_distance(z).value
    
    proj_D1D2 = np.zeros(len(r_bins)) #pair count storage array
    proj_D1R = np.zeros(len(r_bins)) #pair count storage array
    N1 = len(sample1)
    for j in range(0,N1):
        r_proj = X[j]/(1.0+z[j])*np.radians(theta_bins)
        k_ind = np.searchsorted(r_bins,r_proj)-1
        for k in range(0,len(theta_bins)-1):
            #if k_ind[k]<len(r_bins):
            proj_D1D2[k_ind[k]] += D1D2[j,k]
            proj_D1R[k_ind[k]] += D1R[j,k]
    
    proj_D1D2 = proj_D1D2[:-1]
    proj_D1R = proj_D1R[:-1]
    
    if rank==0:
        print(proj_D1D2)
        print(proj_D1R)
    
    
    xi_12 = TP_estimator(proj_D1D2,proj_D1R,None,N1,N2,NR,NR,estimator)
    return xi_12


if __name__ == '__main__':
    main()