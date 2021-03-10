#!/bin/bash

# Run the largest cities first
sbatch analysis_largecity_parsets.job 61
sbatch analysis_largecity_parsets.job 60
sbatch analysis_largecity_parsets.job 59

# Run medium and small cities next
sbatch analysis_mediumcities.job
sbatch analysis_smallcities.job