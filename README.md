# Growing Urban Bicycle Networks

This is the source code for the scientific paper *Growing Urban Bicycle Networks* by M. Szell, S. Mimar, T. Perlman, G. Ghoshal, and R. Sinatra. The code downloads and pre-processes data from OpenStreetMap, prepares points of interest, runs simulations, measures and saves the results, creates videos and plots. Large data files (exports, logs, plots, results, videos) make up many GBs and are stored in the external folder `bikenwgrowth_external` due to Github's space limitations.

Preprint: [insert arXiv link]  
Data repository: [insert link]  
Visualization: [GrowBike.Net](http://growbike.net) (Source code: [https://gitlab.com/Lynghede/bikeviz](https://gitlab.com/Lynghede/bikeviz))


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