#!/usr/bin/python

#Author: Duncan Campbell
#Written: August 12, 2013
#Yale University
#Description: make a random catalogue for each tile in the cfhtlens survey

###packages###
import numpy as np
import os
import fnmatch
import sys
import custom_utilities as cu


def main():
    filepath = cu.get_output_path() + 'analysis/cfhtlens/'
    savepath = cu.get_output_path() + 'analysis/cfhtlens/'

    field1=sys.argv[1]
    field=field1.lower()
    print field1, field

    filenames = os.listdir(filepath)
    filenames = fnmatch.filter(filenames, field+'*.npy')

    dtype = [('ra',float),('dec',float),('flag',int),('num_tile',int)]
    randoms = np.recarray((0,), dtype = dtype)

    #open up all the tile random catalogues and combine
    for filename in filenames:
        print filename
        tile_randoms = np.load(filepath+filename)
        randoms = np.hstack((randoms,tile_randoms))

    print 'maximum number of tiles a random falls in:', max(randoms['num_tile'])

    keep = np.where(np.logical_not((randoms['flag']==8192) & (randoms['num_tile']>1)))[0]
    randoms = randoms[keep]
    

    print len(randoms)
    '''
    #where tiles overlap, take a proportional number of randoms
    #2 tiles overlap
    result_2 = np.where(randoms['num_tile']==2)[0] #identify randoms
    ran_int_2 = np.random.random_integers(1,2,len(result_2)) #roll a dice to keep it
    keep_2 = np.where(ran_int_2==1)[0] #keep 1/(N tiles) of the points
    keep_2 = result_2[keep_2]
    #3 tiles overlap
    result_3 = np.where(randoms['num_tile']==3)[0] #identify randoms
    ran_int_3 = np.random.random_integers(1,3,len(result_3)) #roll a dice to keep it
    keep_3 = np.where(ran_int_3==1)[0] #keep 1/(N tiles) of the points
    keep_3 = result_3[keep_3]
    #4 tiles overlap
    result_4 = np.where(randoms['num_tile']==4)[0] #identify randoms
    ran_int_4 = np.random.random_integers(1,4,len(result_4)) #roll a dice to keep it
    keep_4 = np.where(ran_int_4==1)[0] #keep 1/(N tiles) of the points
    keep_4 = result_4[keep_4]

    #generate master list of randoms to keep
    keep = np.where(randoms['num_tile']==1)[0] #all points in non-pverlap regions
    keep = np.hstack((keep, keep_2, keep_3, keep_4)) #combine the "keep" lists

    randoms = randoms[keep]
    print len(randoms)
    '''

    np.save(savepath+field1+'_randoms', randoms)
    
    
if __name__ == '__main__':
    main()
