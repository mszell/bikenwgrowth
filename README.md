# Growing Urban Bicycle Networks

This is the source code for the scientific paper [*Growing Urban Bicycle Networks*](https://arxiv.org/abs/2107.02185) by [M. Szell](http://michael.szell.net/), S. Mimar, T. Perlman, [G. Ghoshal](http://gghoshal.pas.rochester.edu/), and [R. Sinatra](http://www.robertasinatra.com/). The code downloads and pre-processes data from OpenStreetMap, prepares points of interest, runs simulations, measures and saves the results, creates videos and plots. 

**Preprint**: [arXiv:2107.02185](https://arxiv.org/abs/2107.02185)  
**Data repository**: [zenodo.5083049](https://zenodo.org/record/5083049)  
**Visualization**: [GrowBike.Net](http://growbike.net)  
**Videos & Plots**: [http://growbike.net/download](http://growbike.net/download)

[![Growing Urban Bicycle Networks](readmevideo.gif)](http://growbike.net/city/paris)

## Folder structure
The main folder/repo is `bikenwgrowth`, containing Jupyter notebooks (`code/`), preprocessed data (`data/`), parameters (`parameters/`), result plots (`plots/`), HPC server scripts and jobs (`scripts/`).

Other data files (network plots, videos, results, exports, logs) make up many GBs and are stored in the separate external folder `bikenwgrowth_external` due to Github's space limitations.

## Setting up code environment
```
conda create --override-channels -c conda-forge -n OSMNX python=3 osmnx=0.16.2 python-igraph watermark haversine rasterio tqdm geojson
conda activate OSMNX
conda install -c conda-forge ipywidgets
pip install opencv-python
conda install -c anaconda gdal
pip install --user ipykernel
python -m ipykernel install --user --name=OSMNX
```
Run Jupyter Notebook with kernel OSMNX (Kernel > Change Kernel > OSMNX)

## Running the code on an HPC cluster with SLURM
For multiple, esp. large, cities, running the code on a high performance computing cluster is strongly suggested as the tasks are easy to paralellize. The shell scripts are written for [SLURM](https://slurm.schedmd.com/overview.html).  

1. Populate `parameters/cities.csv`, see below.
2. Run 01 and 02 once locally to download and prepare all networks and POIs (The alternative is server-side `sbatch scripts/download.job`, but OSMNX throws too many connection issues, so manual supervision is needed)
3. Upload `code/*.py`, `parameters/*`, `scripts/*`
4. Run: `./mastersbatch_analysis.sh`
5. Run, if needed: `./mastersbatch_export.sh`
6. After all is finished, run: `./cleanup.sh`
7. Recommended, run: `./fixresults.sh` (to clean up results in case of amended data from repeated runs)

## Running the code locally
Single (or few/small) cities could be run locally but require manual, step-by-step execution of Jupyter notebooks:

1. Populate `parameters/cities.csv`, see below.
2. Run 01 and 02 once to download and prepare all networks and POIs  
3. Run 03,04,05 for each parameter set (see below), set in `parameters/parameters.py`  
4. Run 06 or other steps as needed.

### Parameter sets 
1. `prune_measure = "betweenness"`, `poi_source =  "railwaystation"`  
2. `prune_measure = "betweenness"`, `poi_source =  "grid"`  
3. `prune_measure = "closeness"`, `poi_source =  "railwaystation"`  
4. `prune_measure = "closeness"`, `poi_source =  "grid"`  
5. `prune_measure = "random"`, `poi_source =  "railwaystation"`  
6. `prune_measure = "random"`, `poi_source =  "grid"` 

## Populating cities.csv
### Checking nominatimstring  
* Go to e.g. [https://nominatim.openstreetmap.org/search.php?q=paris%2C+france&polygon_geojson=1&viewbox=](https://nominatim.openstreetmap.org/search.php?q=paris%2C+france&polygon_geojson=1&viewbox=) and enter the search string. If a correct polygon (or multipolygon) pops up it should be fine. If not leave the field empty and acquire a shape file, see below.

### Acquiring shape file  
* Go to [Overpass](overpass-turbo.eu), to the city, and run:
    `relation["boundary"="administrative"]["name:en"="Copenhagen Municipality"]({{bbox}});(._;>;);out skel;`
* Export: Download as GPX
* Use QGIS to create a polygon, with Vector > Join Multiple Lines, and Processing Toolbox > Polygonize (see [Stackexchange answer 1](https://gis.stackexchange.com/questions/98320/connecting-two-line-ends-in-qgis-without-resorting-to-other-software) and [Stackexchange answer 2](https://gis.stackexchange.com/questions/207463/convert-a-line-to-polygon))
