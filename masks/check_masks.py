#!/usr/bin/python

#Author: Duncan Campbell
#Written: September 8, 2014
#Yale University
#Description: check object mask value in cfhtlens catalogue to the fits mask file.

from __future__ import print_function, division
import numpy as np
import custom_utilities as cu

def main():
    """
    read in cfhtlens catalogue and check object mask values against fits mask.
    """
    import h5py
    
    if len(sys.argv)>1:
        field = sys.argv[1]
    else: field='W3'
    if len(sys.argv)>2:
        N_test = sys.argv[2]
    else: N_test = 100
    
    #import cfhtlens catalogues
    filepath = cu.get_output_path()+'processed_data/CFHTLens/'
    f = h5py.File(filepath+'catalogues/'+field+'.hdf5', 'r')
    W = f.get(field)
    
    for i in range(0,N_test)
        ind = np.random.randint(0,len(W),1)
        field = W['field'][ind]
        object_mask_value = W['MASK'][ind]
        
        mask_filename = field+'_izrgu_finalmask_mosaic.fits'
        filepath = cu.get_data_path()+'CFHTLens/masks/'
        data, w = read_mask(filepath, mask_filename)
        
        x,y = get_pix((W['ALPHA_J2000'],W['DELTA_J2000']),w)
        fits_mask_value = data[y,x] #row,column ---> y,x
        print(i,object_mask_value,fits_mask_value)
        assert object_mask_value==fits_mask_value, "fits mask does not match catalogue mask!"


def get_pixel(coords,w):
    """
    get index of pixel
    """
    
    pix = w.wcs_world2pix(coords, 0)
    pix = pix.astype(int)
    
    return pix

def mask_filenames(field):
    """
    return the list of filenames of cfhtlens fits masks for a specified field.
    """
    import os
    import fnmatch

    filenames = os.listdir(cu.get_data_path()+'CFHTLens/masks/')
    filenames = fnmatch.filter(filenames, field+'*.fits')
    
    return filenames


def read_mask(filepath,filename):
    """
    read in cfhtlens fits mask file.
    """
    from astropy.io import fits
    from astropy import wcs
    
    hdulist = fits.open(filepath,filename, memmap=True)
    data = hdulist[0].data
    header = hdulist[0].header
    nxpix = header['NAXIS1']
    nypix = header['NAXIS2']
    center_xpix = header['CRPIX1']
    center_ypix = header['CRPIX2']
    tile  = header['OBJECT']
    w = wcs.WCS(hdulist[0].header)
    
    return data, w


if __name__ == '__main__':
    main()