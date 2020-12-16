# PARAMETERS
# These are values to loop through for different runs
poi_source = "railwaystation" # railwaystation, grid
prune_measure = "betweenness" # betweenness, closeness




# SEMI-CONSTANTS
# These values should not be changed, unless the analysis shows we need to

prune_measures = {"betweenness": "Bq", "closeness": "Cq"}
prune_quantiles = [x/40 for x in list(range(1, 41))] # The quantiles where the GT should be pruned using the prune_measure
networktypes = ["biketrack", "carall", "bikeable", "biketrackcarall"] # Existing infrastructures to analyze

# 02
gridl = 1707 # in m, for generating the grid
# https://en.wikipedia.org/wiki/Right_triangle#Circumcircle_and_incircle
# 2*0.5 = a+a-sqrt(2)a   |   1 = a(2-sqrt2)   |   a = 1/(2-sqrt2) = 1.707
# This leads to a full 500m coverage when a (worst-case) square is being triangulated
bearingbins = 72 # number of bins to determine bearing. e.g. 72 will create 5 degrees bins
poiparameters = {"railwaystation":{'railway':['station','halt']}#,
                 #"busstop":{'highway':'bus_stop'}
                }

# 04
buffer_walk = 500 # Buffer in m for coverage calculations. (How far people are willing to walk)
numnodepairs = 500 # Number of node pairs to consider for random sample to calculate directness (O(numnodepairs^2), so better not go over 1000)

#05
nodesize_grown = 10.5
plotparam = {"bbox": (1280,1280),
			"dpi": 96,
			"carall": {"width": 0.5, "edge_color": '#999999'},
			"biketrack": {"width": 1.5, "edge_color": '#2222ff'},
			"biketrack_offstreet": {"width": 0.75, "edge_color": '#00aa22'},
			"bikeable": {"width": 0.75, "edge_color": '#222222'},
			"bikegrown": {"width": 4.25, "edge_color": '#008ecc', "node_color": '#008ecc'},
			"highlight": {"width": 4.25, "edge_color": '#ff00aa', "node_color": '#ff00aa'},
			"poi_unreached": {"node_color": '#ff7338', "edgecolors": '#ffefe9'},
			"poi_reached": {"node_color": '#004c6c', "edgecolors": '#f1fbff'},
			"abstract": {"edge_color": '#000000', "alpha": 0.75}
			}



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
