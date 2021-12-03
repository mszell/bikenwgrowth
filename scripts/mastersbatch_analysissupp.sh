#!/bin/bash

# Run the largest cities first
sbatch analysissupp_largecity_parsets.job 61
sbatch analysissupp_largecity_parsets.job 60
sbatch analysissupp_largecity_parsets.job 59

# Run medium and small cities next
sbatch analysissupp_mediumcities.job
sbatch analysissupp_smallcities.job