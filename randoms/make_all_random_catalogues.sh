#!/bin/bash

#run random tile catalogue generator for each field
python make_cfhtlens_random_cat.py W1 &
wait
python make_cfhtlens_random_cat.py W2 &
wait
python make_cfhtlens_random_cat.py W3 &
wait
python make_cfhtlens_random_cat.py W4 &

#after all of those are complete, clean up the filenames
wait
sh /scratch/dac29/output/analysis/cfhtlens/clean_up_filenames.sh &

#now, combine each tile catalogue into field catalogues
wait
python combine_cfhtlens_random_cats.py W1 &
wait
python combine_cfhtlens_random_cats.py W2 &
wait
python combine_cfhtlens_random_cats.py W3 &
wait
python combine_cfhtlens_random_cats.py W4 &