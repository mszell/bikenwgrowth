#!/bin/bash

# Run the largest cities first
sbatch scripts/array_largecity_parset.job 61
sbatch scripts/array_largecity_parset.job 60
sbatch scripts/array_largecity_parset.job 59

# Run medium and small cities next
sbatch scripts/array_mediumcities.job
sbatch scripts/array_smallcities.job

# Clean up
./cleanup.sh