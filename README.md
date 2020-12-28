# Algorithmic bicycle network design
#### Michael Szell, Tyler Perlman, Sayat Mimar, Gourab Ghoshal, Roberta Sinatra

## Setting up code environment
```
conda create --override-channels -c   
conda-forge -n OSMNX python=3 osmnx python-igraph watermark haversine rasterio tqdm
conda activate OSMNX
conda install -c conda-forge ipywidgets
pip install opencv-python
conda install -c anaconda gdal
pip install --user ipykernel
python -m ipykernel install --user --name=OSMNX
```
Run Jupyter Notebook with kernel OSMNX (Kernel > Change Kernel > OSMNX)

## Running the code
1. Populate `parameters/cities.csv` 
2. Run 01 and 02 once to download and prepare all networks and POIs  
3. Run 03,04,05 for each parameter set (see below), set in `parameters/parameters.py`  
4. Run 06 for selected cities

### Parameter sets 
`prune_measure = "betweenness"`, `poi_source =  "railwaystation"`  
`prune_measure = "betweenness"`, `poi_source =  "grid"`  
`prune_measure = "closeness"`, `poi_source =  "railwaystation"`  
`prune_measure = "closeness"`, `poi_source =  "grid"`  
