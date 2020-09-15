# Algorithmic bicycle network design
#### Michael Szell, Tyler Perlman, Sayat Mimar, Gourab Ghoshal, Roberta Sinatra

## Running the code
1. Populate `parameters/cities.csv` 
2. Run 01 and 02 to download and prepare all networks and POIs  
3. Choose a parameter set in `parameters/parameters.py`  
4. Run 03
5. Go to 3. until you have finished with all parameter sets
6. Choose a parameter set in `parameters/parameters.py`  
7. Run 04
8. Go to 6. until you have finished with all parameter sets  
6. Run 05, 06 for the different 04 parameter sets  

### All suggested parameter sets 
#### For `03_poi_based_generation`
`prune_measure = "betweenness"`, `poi_source =  "railwaystation"`  
`prune_measure = "betweenness"`, `poi_source =  "grid"`  
`prune_measure = "closeness"`, `poi_source =  "railwaystation"`  
`prune_measure = "closeness"`, `poi_source =  "grid"`  

#### For `04_connect_clusters`
`prune_measure = "betweenness"`, `cutofftype = "abs"`, `cutoff = 1000`  
`prune_measure = "betweenness"`, `cutofftype = "abs"`, `cutoff = 2000`  
`prune_measure = "betweenness"`, `cutofftype = "rel"`, `cutoff = 0.5`  
`prune_measure = "betweenness"`, `cutofftype = "rel"`, `cutoff = 0.8`  
`prune_measure = "closeness"`, `cutofftype = "abs"`, `cutoff = 1000`  
`prune_measure = "closeness"`, `cutofftype = "abs"`, `cutoff = 2000`  
`prune_measure = "closeness"`, `cutofftype = "rel"`, `cutoff = 0.5`  
`prune_measure = "closeness"`, `cutofftype = "rel"`, `cutoff = 0.8`  