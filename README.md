# Algorithmic bicycle network design
#### Michael Szell, Tyler Perlman, Sayat Mimar, Gourab Ghoshal, Roberta Sinatra

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

## Running the code on cluster (using SLURM)
1. Populate `parameters/cities.csv` 
2. Run 01 and 02 once locally to download and prepare all networks and POIs (The alternative is server-side `sbatch scripts/download.job`, but OSMNX throws too many connection issues, so manual supervision is needed)
3. Upload `code/*.py`, `parameters/*`, `scripts/*`
4. Run: `./mastersbatch_analysis.sh`
5. Run, if needed: `./mastersbatch_export.sh`
5. Run: `./cleanup.sh`

## Running the code locally
1. Populate `parameters/cities.csv` 
2. Run 01 and 02 once to download and prepare all networks and POIs  
3. Run 03,04,05 for each parameter set (see below), set in `parameters/parameters.py`  
4. Run 06 for selected cities

### Parameter sets 
1. `prune_measure = "betweenness"`, `poi_source =  "railwaystation"`  
2. `prune_measure = "betweenness"`, `poi_source =  "grid"`  
3. `prune_measure = "closeness"`, `poi_source =  "railwaystation"`  
4. `prune_measure = "closeness"`, `poi_source =  "grid"`  
5. `prune_measure = "random"`, `poi_source =  "railwaystation"`  
6. `prune_measure = "random"`, `poi_source =  "grid"` 