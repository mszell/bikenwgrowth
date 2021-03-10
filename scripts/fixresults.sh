#!/bin/bash
# This script takes from all poi+metric results csv files the header and the last 40 rows,
# fixing the problem of appended results when SLURM jobs of same cities were repeated

# https://stackoverflow.com/questions/9612090/how-to-loop-through-file-names-returned-by-find
#shopt -s globstar
for f in ../../bikenwgrowth_external/results/**/*ness.csv
do
    echo $f
    cp $f tmp.csv
    head -1 tmp.csv > $f
    tail -40 tmp.csv >> $f
done

for f in ../../bikenwgrowth_external/results/**/*_random.csv
do
    echo $f
    cp $f tmp.csv
    head -1 tmp.csv > $f
    tail -40 tmp.csv >> $f
done

rm tmp.csv