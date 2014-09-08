#!/usr/bin/python

#Author: Duncan Campbell
#Written: August 12, 2013
#Yale University
#Description: make a random catalogue for each tile in the cfhtlens survey.  Each tile has
#a fixed angular number density.

###packages###
import numpy as np
import custom_utilities as cu
from CFHTLens import inside_cfhtlens
from astropy.io import fits
from astropy import wcs
from multiprocessing import Pool
import os
import fnmatch
import sys


def main():
    global filepath
    global savepath
    filepath = cu.get_data_path() + 'CFHTLens/masks/'
    savepath = cu.get_output_path() + 'analysis/cfhtlens/'

    field = sys.argv[1]
    print 'field:', field

    if len(sys.argv)>2:
        filenames = sys.argv[2]
    else:
        filenames = os.listdir(cu.get_data_path()+'CFHTLens/masks/')
        filenames = fnmatch.filter(filenames, field+'*.fits')
    # a "?" as the 2nd user input returns the lost of mask files and exits.
    if filenames == '?':
        filenames = os.listdir(cu.get_data_path()+'CFHTLens/masks/')
        filenames = fnmatch.filter(filenames, field+'*.fits')
        for filename in filenames: print(filename)
        return 0

    p = Pool(8)
    p.map(do_work,filenames)


def do_work(filename):
        print filename
        hdulist = fits.open(filepath+filename, memmap=True)
        data = hdulist[0].data
        header = hdulist[0].header
        nxpix = header['NAXIS1']
        nypix = header['NAXIS2']
        center_xpix = header['CRPIX1']
        center_ypix = header['CRPIX2']
        tile  = header['OBJECT']
        w = wcs.WCS(hdulist[0].header)

        #randomly sample the tile area
        N_points = 1000000
        n_points = 2.0*N_points
        center_world = w.wcs_pix2world([[center_xpix,center_ypix]], 1)
        center_ra = center_world[0][0]
        center_dec = center_world[0][1]
        ran_coords = cu.sample_spherical_cap(center_ra, center_dec, 0.75, n_points)
        center_ra = center_ra*np.ones(n_points) #convert to array to pass through map
        center_dec = center_dec*np.ones(n_points) #convert to array to pass through map
        ran_ra = np.array(zip(*ran_coords)[0])
        ran_dec = np.array(zip(*ran_coords)[1])
        ran_coords = np.array(ran_coords)

        #determine if the points are inside the tile area
        result  = np.array(map(inside_cfhtlens.within_tile, ran_ra, ran_dec, center_ra, center_dec))
        #trim down the sample to the desired number of points
        #!!!!! don't do this.  you want a constant angular number density, not N inside the tile.
        #ran_ra = ran_ra[result][0:N_points]
        #ran_dec = ran_dec[result][0:N_points]
        #ran_coords = ran_coords[result][0:N_points]
        ran_ra = ran_ra[result]
        ran_dec = ran_dec[result]
        ran_coords = ran_coords[result]
        #determine how many tiles the point is within(overlap regions)
        num_tile = np.array(map(inside_cfhtlens.num_tile, ran_ra, ran_dec))

        #find pixel locations of random points
        ran_coords = np.array(ran_coords)
        ran_pix = w.wcs_world2pix(ran_coords, 0)
        ran_pix = ran_pix.astype(int)
        ran_xpix = np.array(zip(*ran_pix)[0])
        ran_ypix = np.array(zip(*ran_pix)[1])

        #create an array and save random catalogue
        dtype = [('ra',float),('dec',float),('flag',int),('num_tile',int)]
        randoms = np.recarray((len(ran_ra),), dtype = dtype)
        randoms['ra'] = ran_ra
        randoms['dec'] = ran_dec
        randoms['flag'] = data[ran_ypix, ran_xpix] #row,column ---> y,x
        randoms['num_tile'] = num_tile

        np.save(savepath+tile+'_randoms', randoms)

        hdulist.close()
        return 0



if __name__ == '__main__':
    main()
