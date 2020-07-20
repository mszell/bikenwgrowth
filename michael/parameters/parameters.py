# PARAMETERS
# These are values to loop through for different runs

# 03, 05, 06
poi_source = "railwaystation" # railwaystation, grid

# 03, 04, 05, 06
prune_measure = "betweenness" # betweenness, closeness

# 04, 05, 06
cutofftype = "abs" # abs, rel
# Case rel: cutoff (0-1) is fraction threshold of total length
# Case abs: cutoff (in meters) is minimal length of cluster to be considered
cutoff = 2000 # 0.5, 0.8, 1000, 2000




# SEMI-CONSTANTS
# These values should not be changed, unless the analysis shows we need to

# 02
gridl = 1707 # in m, for generating the grid
# https://en.wikipedia.org/wiki/Right_triangle#Circumcircle_and_incircle
# 2*0.5 = a+a-sqrt(2)a   |   1 = a(2-sqrt2)   |   a = 1/(2-sqrt2) = 1.707
# This leads to a full 500m coverage when a (worst-case) square is being triangulated
bearingbins = 72 # number of bins to determine bearing. e.g. 72 will create 5 degrees bins
poiparameters = {"railwaystation":{'railway':['station','halt']}#,
                 #"busstop":{'highway':'bus_stop'}
                }

# 05
buffer_walk = 500 # Buffer in m for coverage calculations. (How far people are willing to walk)
numnodepairs = 10 # Number of node pairs to consider for random sample to calculate directness (O(numnodepairs^2), so better not go over 1000)
networktypes = ["biketrack", "carall", "bikeable", "biketrackcarall"] # Existing infrastructures to analyze

# 03, 04, 05, 06
prune_measures = {"betweenness": "Bq", "closeness": "Cq"}
prune_quantiles = [x/40 for x in list(range(1, 41))] # The quantiles where the GT should be pruned using the prune_measure




# CONSTANTS
# These values should be set once and not be changed

# 01
osmnxparameters = {'car30': {'network_type':'drive', 'custom_filter':'["maxspeed"~"^30$|^20$|^15$|^10$|^5$|^20 mph|^15 mph|^10 mph|^5 mph"]', 'export': True},
                   'carall': {'network_type':'drive', 'custom_filter':'', 'export': True},
                   'bike_cyclewaytrack': {'network_type':'bike', 'custom_filter':'["cycleway"~"track"]', 'export': False},
                   'bike_highwaycycleway': {'network_type':'bike', 'custom_filter':'["highway"~"cycleway"]', 'export': False},
                   'bike_bicycledesignated': {'network_type':'bike', 'custom_filter':'["bicycle"~"designated"]', 'export': False},
                   'bike_cyclewayrighttrack': {'network_type':'bike', 'custom_filter':'["cycleway:right"~"track"]', 'export': False},
                   'bike_cyclewaylefttrack': {'network_type':'bike', 'custom_filter':'["cycleway:left"~"track"]', 'export': False}
                  }  
# Special case 'biketrack': "cycleway"~"track" OR "highway"~"cycleway" OR "bicycle"~"designated" OR "cycleway:right=track" OR "cycleway:left=track"
# Special case 'bikeable': biketrack OR car30
# See: https://wiki.openstreetmap.org/wiki/Key:cycleway#Cycle_tracks

# 02
snapthreshold = 500 # in m, tolerance for snapping POIs to network
consolidatethreshold = 15  # in m, tolerance for consolidating intersections

print("Loaded parameters.\n")
