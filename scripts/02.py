# Load all carall graphs in OSMNX format
G_caralls = {}
G_caralls_simplified = {}
locations = {}
parameterinfo = osmnxparameters['carall']

for placeid, placeinfo in tqdm(cities.items(), desc = "Cities"):
    print(placeid + ": Loading location polygon and carall graph")
    
    if placeinfo["nominatimstring"] != '':
        location = ox.geocoder.geocode_to_gdf(placeinfo["nominatimstring"])
        location = fill_holes(extract_relevant_polygon(placeid, shapely.geometry.shape(location['geometry'][0])))
    else:
        # https://gis.stackexchange.com/questions/113799/how-to-read-a-shapefile-in-python
        shp = fiona.open(PATH["data"] + placeid + "/" + placeid + ".shp")
        first = next(iter(shp))
        location = shapely.geometry.shape(first['geometry'])
    locations[placeid] = location
    
    G_caralls[placeid] = csv_to_ox(PATH["data"] + placeid + "/", placeid, 'carall')
    G_caralls[placeid].graph["crs"] = 'epsg:4326' # needed for OSMNX's graph_to_gdfs in utils_graph.py
    G_caralls_simplified[placeid] = csv_to_ox(PATH["data"] + placeid + "/", placeid, 'carall_simplified')
    G_caralls_simplified[placeid].graph["crs"] = 'epsg:4326' # needed for OSMNX's graph_to_gdfs in utils_graph.py


# Here POIs are downloaded and matched to the network. To ensure consistency, we should download POIs only once, then load them locally. For now we leave it as is, as POIs are not expected to change fast.

for placeid, placeinfo in tqdm(cities.items(), desc = "Cities"):
    print(placeid + ": Creating POIs")
    
    # We need the carall graph and location geometry
    location = locations[placeid]
    G_carall = G_caralls_simplified[placeid]
    
    for poiid, poitag in poiparameters.items():
        gdf = ox.geometries.geometries_from_polygon(location, poitag)
        gdf = gdf[gdf['geometry'].type == "Point"] # only consider points, no polygons etc
        # Now snap to closest nodes in street network, save the nearest node ids
        nnids = set()
        for g in gdf['geometry']:
            n = ox.distance.get_nearest_node(G_carall, [g.y, g.x])
            if not n in nnids and haversine((g.y, g.x), (G_carall.nodes[n]["y"], G_carall.nodes[n]["x"]), unit="m") <= snapthreshold:
                nnids.add(n)
#         nnids = ox.distance.get_nearest_nodes(G_carall, gdf['geometry'].x, gdf['geometry'].y) # This could be faster but does not seem to work correctly
        with open(PATH["data"] + placeid + "/" + placeid + '_' + 'poi_' + poiid + '_nnidscarall.csv', 'w') as f:
            for item in nnids:
                f.write("%s\n" % item)
        
        gdf = gdf.apply(lambda c: c.astype(str) if c.name != 'geometry' else c, axis=0)
        try: # For some cities writing the gdf does not work (i.e. London, Manhattan)
            gdf.to_file(PATH["data"] + placeid + "/" + placeid + '_' + 'poi_' + poiid + '.gpkg', driver = 'GPKG')
        except:
            print("Notice: Writing the gdf did not work for " + placeid)
        if debug: gdf.plot(color = 'red')


for placeid, placeinfo in tqdm(cities.items(), desc  = "Cities"):
    print(placeid + ": Creating grid")
    
    location = locations[placeid]
    
    
    # FIRST, determine the most common bearing, for the best grid orientation
    G = G_caralls[placeid]
    bearings = {}    
    # calculate edge bearings
    Gu = ox.bearing.add_edge_bearings(ox.get_undirected(G))

    # weight bearings by length (meters)
    city_bearings = []
    for u, v, k, d in Gu.edges(keys = True, data = True):
        city_bearings.extend([d['bearing']] * int(d['length']))
    b = pd.Series(city_bearings)
    bearings = pd.concat([b, b.map(reverse_bearing)]).reset_index(drop = 'True')

    bins = np.arange(bearingbins + 1) * 360 / bearingbins
    count = count_and_merge(bearingbins, bearings)
    principalbearing = bins[np.where(count == max(count))][0]
    if debug: 
        print("Principal bearing: " + str(principalbearing))


    # SECOND, construct the grid
    G = G_caralls_simplified[placeid]

    # 1) Get lat lon window, with buffer for snapping outside POIs
    # https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters
    buf = max(((2*snapthreshold) / 6378000) * (180 / math.pi), 
              ((2*snapthreshold) / 6378000) * (180 / math.pi) / math.cos(location.centroid.y * math.pi/180)
             )
    cities[placeid]["bbox"] = location.buffer(buf).bounds

    # 2) Generate abstract grid points in window
    # https://stackoverflow.com/questions/40342355/how-can-i-generate-a-regular-geographic-grid-using-python
    # Set up projections
    p_ll = pyproj.Proj('+proj=longlat +datum=WGS84')
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    p_mt = pyproj.Proj(aeqd_proj.format(lat=location.centroid.y, lon=location.centroid.x)) # equidistant projection

    # Make the area larger to account for tilted grid
    deltax = cities[placeid]["bbox"][2] - cities[placeid]["bbox"][0]
    deltay = cities[placeid]["bbox"][3] - cities[placeid]["bbox"][1]
    enlargefactor = 10

    # Create corners of rectangle to be transformed to a grid
    sw = shapely.geometry.Point((cities[placeid]["bbox"][0], cities[placeid]["bbox"][1]))
    ne = shapely.geometry.Point((cities[placeid]["bbox"][2]+enlargefactor*deltax, cities[placeid]["bbox"][3]+enlargefactor*deltay))
    
    # Project corners to target projection
    transformed_sw = pyproj.transform(p_ll, p_mt, sw.x, sw.y) # Transform NW point to equidistant
    transformed_ne = pyproj.transform(p_ll, p_mt, ne.x, ne.y) # .. same for SE

    # Iterate over 2D area
    principalbearing = principalbearing % 90 # Take it modulo 90 because it will be a square grid
    if principalbearing > 45:
        principalbearing -= 90 # Make bearing fall btw -45 and 45

    xcoords = np.arange(transformed_sw[0], transformed_ne[0], gridl)
    ycoords = np.arange(transformed_sw[1], transformed_ne[1], gridl)
    xsize =  xcoords.size
    ysize = ycoords.size
    xcoords = np.tile(xcoords, ysize)
    ycoords = np.repeat(ycoords, xsize)
    gridpoints=[(x, y) for x, y in zip(xcoords, ycoords)]
    new_points = rotate_grid(gridpoints, origin = transformed_sw, degrees = principalbearing)
    
    # https://stackoverflow.com/questions/42459068/projecting-a-numpy-array-of-coordinates-using-pyproj
    fx, fy = pyproj.transform(p_mt, p_ll, new_points[:,0], new_points[:,1])
    gridpoints = np.dstack([fx, fy])[0]
    if principalbearing >=0:
        # If we rotated right, we need to shift everything to the left
        gridpoints[:,0] -= 0.4*enlargefactor*deltax*math.sin(np.deg2rad(principalbearing))
    else:
        # If we rotated left, we need to shift everything down and to the right
        gridpoints[:,0] += 0.4*enlargefactor*deltax*math.sin(np.deg2rad(principalbearing))
        gridpoints[:,1] -= 0.4*enlargefactor*deltay

    # Cut back to bounding box
    mask = (gridpoints[:,0] >= cities[placeid]["bbox"][0]) & (gridpoints[:,0] <= cities[placeid]["bbox"][2]) & (gridpoints[:,1] >= cities[placeid]["bbox"][1]) & (gridpoints[:,1] <= cities[placeid]["bbox"][3])
    gridpoints_cut = gridpoints[mask]
    
    if debug:
        fig = plt.figure(figsize=[2*6.4, 2*4.8])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_aspect('equal', adjustable = 'datalim')
        plt.plot([g[0] for g in gridpoints_cut], [g[1] for g in gridpoints_cut], ".", color = "red")

    # 3) Snap grid points to map
    nnids = set()
    for g in gridpoints_cut:
        n = ox.distance.get_nearest_node(G, [g[1], g[0]])
        if n not in nnids and haversine((g[1], g[0]), (G.nodes[n]["y"], G.nodes[n]["x"]), unit="m") <= snapthreshold:
            nnids.add(n)
    with open(PATH["data"] + placeid + "/" + placeid + '_poi_grid_nnidscarall.csv', 'w') as f:
        for item in nnids:
            f.write("%s\n" % item)