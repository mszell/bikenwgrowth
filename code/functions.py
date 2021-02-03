
# GRAPH PLOTTING

def holepatchlist_from_cov(cov, map_center):
    """Get a patchlist of holes (= shapely interiors) from a coverage Polygon or MultiPolygon
    """
    holeseq_per_poly = get_holes(cov)
    holepatchlist = []
    for hole_per_poly in holeseq_per_poly:
        for hole in hole_per_poly:
            holepatchlist.append(hole_to_patch(hole, map_center))
    return holepatchlist

def fill_holes(cov):
    """Fill holes (= shapely interiors) from a coverage Polygon or MultiPolygon
    """
    holeseq_per_poly = get_holes(cov)
    holes = []
    for hole_per_poly in holeseq_per_poly:
        for hole in hole_per_poly:
            holes.append(hole)
    eps = 0.00000001
    if isinstance(cov, shapely.geometry.multipolygon.MultiPolygon):
        cov_filled = ops.unary_union([poly for poly in cov] + [Polygon(hole).buffer(eps) for hole in holes])
    elif isinstance(cov, shapely.geometry.polygon.Polygon) and not cov.is_empty:
        cov_filled = ops.unary_union([cov] + [Polygon(hole).buffer(eps) for hole in holes])
    return cov_filled

def extract_relevant_polygon(placeid, mp):
    """Return the most relevant polygon of a multipolygon mp, for being considered the city limit.
    Depends on location.
    """
    if isinstance(mp, shapely.geometry.polygon.Polygon):
        return mp
    if placeid == "tokyo": # If Tokyo, take poly with most northern bound, otherwise largest
        p = max(mp, key=lambda a: a.bounds[-1])
    else:
        p = max(mp, key=lambda a: a.area)
    return p

def get_holes(cov):
    """Get holes (= shapely interiors) from a coverage Polygon or MultiPolygon
    """
    holes = []
    if isinstance(cov, shapely.geometry.multipolygon.MultiPolygon):
        for pol in cov: # cov is generally a MultiPolygon, so we iterate through its Polygons
            holes.append(pol.interiors)
    elif isinstance(cov, shapely.geometry.polygon.Polygon) and not cov.is_empty:
        holes.append(cov.interiors)
    return holes

def cov_to_patchlist(cov, map_center, return_holes = True):
    """Turns a coverage Polygon or MultiPolygon into a matplotlib patch list, for plotting
    """
    p = []
    if isinstance(cov, shapely.geometry.multipolygon.MultiPolygon):
        for pol in cov: # cov is generally a MultiPolygon, so we iterate through its Polygons
            p.append(pol_to_patch(pol, map_center))
    elif isinstance(cov, shapely.geometry.polygon.Polygon) and not cov.is_empty:
        p.append(pol_to_patch(cov, map_center))
    if not return_holes:
        return p
    else:
        holepatchlist = holepatchlist_from_cov(cov, map_center)
        return p, holepatchlist

def pol_to_patch(pol, map_center):
    """Turns a coverage Polygon into a matplotlib patch, for plotting
    """
    y, x = pol.exterior.coords.xy
    pos_transformed, _ = project_pos(y, x, map_center)
    return matplotlib.patches.Polygon(pos_transformed)

def hole_to_patch(hole, map_center):
    """Turns a LinearRing (hole) into a matplotlib patch, for plotting
    """
    y, x = hole.coords.xy
    pos_transformed, _ = project_pos(y, x, map_center)
    return matplotlib.patches.Polygon(pos_transformed)


def initplot():
    fig = plt.figure(figsize=(plotparam["bbox"][0]/plotparam["dpi"], plotparam["bbox"][1]/plotparam["dpi"]), dpi=plotparam["dpi"])
    plt.axes().set_aspect('equal')
    plt.axes().set_xmargin(0.01)
    plt.axes().set_ymargin(0.01)
    plt.axes().set_axis_off()
    return fig

def nodesize_from_pois(nnids):
    """Determine POI node size based on number of POIs.
    The more POIs the smaller (linearly) to avoid overlaps.
    """
    minnodesize = 30
    maxnodesize = 220
    return max(minnodesize, maxnodesize-len(nnids))


def simplify_ig(G):
    """Simplify an igraph with ox.simplify_graph
    """
    G_temp = copy.deepcopy(G)
    G_temp.es["length"] = G_temp.es["weight"]
    output = ig.Graph.from_networkx(ox.simplify_graph(nx.MultiDiGraph(G_temp.to_networkx())).to_undirected())
    output.es["weight"] = output.es["length"]
    return output


def nxdraw(G, networktype, map_center = False, nnids = False, drawfunc = "nx.draw", nodesize = 0, weighted = False, maxwidthsquared = 0, simplified = False):
    """Take an igraph graph G and draw it with a networkx drawfunc.
    """
    if simplified:
        G.es["length"] = G.es["weight"]
        G_nx = ox.simplify_graph(nx.MultiDiGraph(G.to_networkx())).to_undirected()
    else:
        G_nx = G.to_networkx()
    if nnids is not False: # Restrict to nnids node ids
        nnids_nx = [k for k,v in dict(G_nx.nodes(data=True)).items() if v['id'] in nnids]
        G_nx = G_nx.subgraph(nnids_nx)
        
    pos_transformed, map_center = project_nxpos(G_nx, map_center)
    if weighted is True:
        # The max width should be the node diameter (=sqrt(nodesize))
        widths = list(nx.get_edge_attributes(G_nx, "weight").values())
        widthfactor = 1.1 * math.sqrt(maxwidthsquared) / max(widths)
        widths = [max(0.33, w * widthfactor) for w in widths]
        eval(drawfunc)(G_nx, pos_transformed, **plotparam[networktype], node_size = nodesize, width = widths)
    elif type(weighted) is float or type(weighted) is int and weighted > 0:
        eval(drawfunc)(G_nx, pos_transformed, **plotparam[networktype], node_size = nodesize, width = weighted)
    else:
        eval(drawfunc)(G_nx, pos_transformed, **plotparam[networktype], node_size = nodesize)
    return map_center



# OTHER FUNCTIONS

def common_entries(*dcts):
    """Like zip() but for dicts.
    See: https://stackoverflow.com/questions/16458340/python-equivalent-of-zip-for-dictionaries
    """
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)

def project_nxpos(G, map_center = False):
    """Take a spatial nx network G and projects its GPS coordinates to local azimuthal.
    Returns transformed positions, as used by nx.draw()
    """
    lats = nx.get_node_attributes(G, 'x')
    lons = nx.get_node_attributes(G, 'y')
    pos = {nid:(lat,-lon) for (nid,lat,lon) in common_entries(lats,lons)}
    if map_center:
        loncenter = map_center[0]
        latcenter = map_center[1]
    else:
        loncenter = np.mean(list(lats.values()))
        latcenter = -1* np.mean(list(lons.values()))
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(latcenter, loncenter)
    # Use transformer: https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    wgs84_to_aeqd = pyproj.Transformer.from_proj(
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))
    pos_transformed = {nid:list(ops.transform(wgs84_to_aeqd.transform, Point(latlon)).coords)[0] for nid, latlon in pos.items()}
    return pos_transformed, (loncenter,latcenter)


def project_pos(lats, lons, map_center = False):
    """Project GPS coordinates to local azimuthal.
    """
    pos = [(lat,-lon) for lat,lon in zip(lats,lons)]
    if map_center:
        loncenter = map_center[0]
        latcenter = map_center[1]
    else:
        loncenter = np.mean(list(lats.values()))
        latcenter = -1* np.mean(list(lons.values()))
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(latcenter, loncenter)
    # Use transformer: https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    wgs84_to_aeqd = pyproj.Transformer.from_proj(
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))
    pos_transformed = [(ops.transform(wgs84_to_aeqd.transform, Point(latlon)).coords)[0] for latlon in pos]
    return pos_transformed, (loncenter,latcenter)


def round_coordinates(G, r = 7):
    for v in G.vs:
        G.vs[v.index]["x"] = round(G.vs[v.index]["x"], r)
        G.vs[v.index]["y"] = round(G.vs[v.index]["y"], r)

def mirror_y(G):
    for v in G.vs:
        y = G.vs[v.index]["y"]
        G.vs[v.index]["y"] = -y
    
def dist(v1,v2):
    dist = haversine((v1['x'],v1['y']),(v2['x'],v2['y']))
    return dist


def osm_to_ig(node, edge):
    """ Turns a node and edge dataframe into an igraph Graph.
    """
    
    G = ig.Graph(directed = False)

    x_coords = node['x'].tolist() 
    y_coords = node['y'].tolist()
    ids = node['osmid'].tolist()
    coords = []

    for i in range(len(x_coords)):
        G.add_vertex(x = x_coords[i], y = y_coords[i], id = ids[i])
        coords.append((x_coords[i], y_coords[i]))

    id_dict = dict(zip(G.vs['id'], np.arange(0, G.vcount()).tolist()))
    coords_dict = dict(zip(np.arange(0, G.vcount()).tolist(), coords))

    edge_list = []
    for i in range(len(edge)):
        edge_list.append([id_dict.get(edge['u'][i]), id_dict.get(edge['v'][i])])
        
    G.add_edges(edge_list)
    G.simplify()
    new_edges = G.get_edgelist()
    
    distances_list = []
    for i in range(len(new_edges)):
        distances_list.append(haversine(coords_dict.get(new_edges[i][0]), coords_dict.get(new_edges[i][1])))

    G.es()['weight'] = distances_list
    
    return G

def compress_file(p, f, filetype = ".csv", delete_uncompressed = True):
    with zipfile.ZipFile(p + f + ".zip", 'w', zipfile.ZIP_DEFLATED) as zfile:
        zfile.write(p + f + filetype, f + filetype)
    if delete_uncompressed: os.remove(p + f + filetype)

def ox_to_csv(G, p, placeid, parameterid, postfix = "", compress = True, verbose = True):
    if "crs" not in G.graph:
        G.graph["crs"] = 'epsg:4326' # needed for OSMNX's graph_to_gdfs in utils_graph.py
    try:
        node, edge = ox.graph_to_gdfs(G)
    except ValueError:
        node, edge = gpd.GeoDataFrame(), gpd.GeoDataFrame()
    prefix = placeid + '_' + parameterid + postfix

    node.to_csv(p + prefix + '_nodes.csv', index = False)
    if compress: compress_file(p, prefix + '_nodes')
 
    edge.to_csv(p + prefix + '_edges.csv', index = False)
    if compress: compress_file(p, prefix + '_edges')

    if verbose: print(placeid + ": Successfully wrote graph " + parameterid + postfix)

def check_extract_zip(p, prefix):
    """ Check if a zip file prefix+'_nodes.zip' and + prefix+'_edges.zip'
    is available at path p. If so extract it and return True, otherwise False.
    If you call this function, remember to clean up (i.e. delete the unzipped files)
    after you are done like this:

    if compress:
        os.remove(p + prefix + '_nodes.csv')
        os.remove(p + prefix + '_edges.csv')
    """

    try: # Use zip files if available
        with zipfile.ZipFile(p + prefix + '_nodes.zip', 'r') as zfile:
            zfile.extract(prefix + '_nodes.csv', p)
        with zipfile.ZipFile(p + prefix + '_edges.zip', 'r') as zfile:
            zfile.extract(prefix + '_edges.csv', p)
        return True
    except:
        return False


def csv_to_ox(p, placeid, parameterid):
    """ Load a networkx graph from _edges.csv and _nodes.csv
    The edge file must have attributes u,v,osmid
    The node file must have attributes y,x,osmid
    Only these attributes are loaded, and edge lengths are calculated.
    """
    prefix = placeid + '_' + parameterid
    compress = check_extract_zip(p, prefix)
    
    with open(p + prefix + '_edges.csv', 'r') as f:
        header = f.readline().strip().split(",")

        lines = []
        for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            line_list = [c for c in line]
            osmid = str(eval(line_list[header.index("osmid")])[0]) if isinstance(eval(line_list[header.index("osmid")]), list) else line_list[header.index("osmid")] # If this is a list due to multiedges, just load the first osmid
            line_string = "" + line_list[header.index("u")] + " "+ line_list[header.index("v")] + " " + osmid
            lines.append(line_string)
        G = nx.parse_edgelist(lines, nodetype = int, data = (("osmid", int),), create_using = nx.MultiDiGraph) # MultiDiGraph is necessary for OSMNX, for example for get_undirected(G) in utils_graph.py
    with open(p + prefix + '_nodes.csv', 'r') as f:
        header = f.readline().strip().split(",")
        values_x = {}
        values_y = {}
        for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            line_list = [c for c in line]
            osmid = int(line_list[header.index("osmid")])
            values_x[osmid] = float(line_list[header.index("x")])
            values_y[osmid] = float(line_list[header.index("y")])

        nx.set_node_attributes(G, values_x, "x")
        nx.set_node_attributes(G, values_y, "y")

    edge_lengths = {}
    for e in G.edges(keys=True):
        edge_lengths[e] = haversine((values_x[e[0]], values_y[e[0]]), (values_x[e[1]], values_y[e[1]]))
    nx.set_edge_attributes(G, edge_lengths, "length")

    if compress:
        os.remove(p + prefix + '_nodes.csv')
        os.remove(p + prefix + '_edges.csv')
    return G

def csv_to_ig(p, placeid, parameterid):
    prefix = placeid + '_' + parameterid
    compress = check_extract_zip(p, prefix)
    empty = False
    try:
        n = pd.read_csv(p + prefix + '_nodes.csv')
        e = pd.read_csv(p + prefix + '_edges.csv')
    except:
        empty = True
    if compress:
        os.remove(p + prefix + '_nodes.csv')
        os.remove(p + prefix + '_edges.csv')
    if empty:
        return ig.Graph(directed = False)
    G = osm_to_ig(n, e)
    round_coordinates(G)
    mirror_y(G)
    return G

def ig_to_geojson(G):
    linestring_list = []
    for e in G.es():
        linestring_list.append(geojson.LineString([(e.source_vertex["x"], -e.source_vertex["y"]), (e.target_vertex["x"], -e.target_vertex["y"])]))
    G_geojson = geojson.GeometryCollection(linestring_list)
    return G_geojson




# NETWORK GENERATION

def highest_closeness_node(G):
    closeness_values = G.closeness(weights = 'weight')
    sorted_closeness = sorted(closeness_values, reverse = True)
    index = closeness_values.index(sorted_closeness[0])
    return G.vs(index)['id']

def clusterindices_by_length(clusterinfo, rev = True):
    return [k for k, v in sorted(clusterinfo.items(), key=lambda item: item[1]["length"], reverse = rev)]

class MyPoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

def segments_intersect(A,B,C,D):
    """Check if two line segments intersect (except for colinearity)
    Returns true if line segments AB and CD intersect properly.
    Adapted from: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    """
    if (A.x == C.x and A.y == C.y) or (A.x == D.x and A.y == D.y) or (B.x == C.x and B.y == C.y) or (B.x == D.x and B.y == D.y): return False # If the segments share an endpoint they do not intersect properly
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def new_edge_intersects(G, enew):
    """Given a graph G and a potential new edge enew,
    check if enew will intersect any old edge.
    """
    E1 = MyPoint(enew[0], enew[1])
    E2 = MyPoint(enew[2], enew[3])
    for e in G.es():
        O1 = MyPoint(e.source_vertex["x"], e.source_vertex["y"])
        O2 = MyPoint(e.target_vertex["x"], e.target_vertex["y"])
        if segments_intersect(E1, E2, O1, O2):
            return True
    return False
    

def delete_overlaps(G_res, G_orig, verbose = False):
    """Deletes inplace all overlaps of G_res with G_orig (from G_res)
    based on node ids. In other words: G_res -= G_orig
    """
    del_edges = []
    for e in list(G_res.es):
        try:
            n1_id = e.source_vertex["id"]
            n2_id = e.target_vertex["id"]
            # If there is already an edge in the original network, delete it
            n1_index = G_orig.vs.find(id = n1_id).index
            n2_index = G_orig.vs.find(id = n2_id).index
            if G_orig.are_connected(n1_index, n2_index):
                del_edges.append(e.index)
        except:
            pass
    G_res.delete_edges(del_edges)
    # Remove isolated nodes
    isolated_nodes = G_res.vs.select(_degree_eq=0)
    G_res.delete_vertices(isolated_nodes)
    if verbose: print("Removed " + str(len(del_edges)) + " overlapping edges and " + str(len(isolated_nodes)) + " nodes.")

def constrict_overlaps(G_res, G_orig, factor = 5):
    """Increases length by factor of all overlaps of G_res with G_orig (in G_res) based on edge ids.
    """
    for e in list(G_res.es):
        try:
            n1_id = e.source_vertex["id"]
            n2_id = e.target_vertex["id"]
            n1_index = G_orig.vs.find(id = n1_id).index
            n2_index = G_orig.vs.find(id = n2_id).index
            if G_orig.are_connected(n1_index, n2_index):
                G_res.es[e.index]["weight"] = factor * G_res.es[e.index]["weight"]
        except:
            pass



    

def greedy_triangulation_routing_clusters(G, G_total, clusters, clusterinfo, prune_quantiles = [1], prune_measure = "betweenness", verbose = False, full_run = False):
    """Greedy Triangulation (GT) of a bike network G's clusters,
    then routing on the graph G_total that includes car infra to connect the GT.
    G and G_total are ipgraph graphs
    
    The GT connects pairs of clusters in ascending order of their distance provided
    that no edge crossing is introduced. It leads to a maximal connected planar
    graph, while minimizing the total length of edges considered. 
    See: cardillo2006spp
    
    Distance here is routing distance, while edge crossing is checked on an abstract 
    level.
    """
    
    if len(clusters) < 2: return ([], []) # We can't do anything with less than 2 clusters

    centroid_indices = [v["centroid_index"] for k, v in sorted(clusterinfo.items(), key=lambda item: item[1]["size"], reverse = True)]
    G_temp = copy.deepcopy(G_total)
    for e in G_temp.es: # delete all edges
        G_temp.es.delete(e)
    
    clusterpairs = clusterpairs_by_distance(G, G_total, clusters, clusterinfo, True, verbose, full_run)
    if len(clusterpairs) == 0: return ([], [])
    
    centroidpairs = [((clusterinfo[c[0][0]]['centroid_id'], clusterinfo[c[0][1]]['centroid_id']), c[2]) for c in clusterpairs]
    
    GT_abstracts = []
    GTs = []
    for prune_quantile in prune_quantiles:
        GT_abstract = copy.deepcopy(G_temp.subgraph(centroid_indices))
        GT_abstract = greedy_triangulation(GT_abstract, centroidpairs, prune_quantile, prune_measure)
        GT_abstracts.append(GT_abstract)

        centroidids_closestnodeids = {} # dict for retrieveing quickly closest node ids pairs from centroidid pairs
        for x in clusterpairs:
            centroidids_closestnodeids[(clusterinfo[x[0][0]]["centroid_id"], clusterinfo[x[0][1]]["centroid_id"])] = (x[1][0], x[1][1])
            centroidids_closestnodeids[(clusterinfo[x[0][1]]["centroid_id"], clusterinfo[x[0][0]]["centroid_id"])] = (x[1][1], x[1][0]) # also add switched version as we do not care about order

        # Get node pairs we need to route, sorted by distance
        routenodepairs = []
        for e in GT_abstract.es:
            # get the centroid-ids from closestnode-ids
            routenodepairs.append([centroidids_closestnodeids[(e.source_vertex["id"], e.target_vertex["id"])], e["weight"]])

        routenodepairs.sort(key=lambda x: x[1])

        # Do the routing, on G_total
        GT_indices = set()
        for poipair, poipair_distance in routenodepairs:
            poipair_ind = (G_total.vs.find(id = poipair[0]).index, G_total.vs.find(id = poipair[1]).index)
            sp = set(G_total.get_shortest_paths(poipair_ind[0], poipair_ind[1], weights = "weight", output = "vpath")[0])
            GT_indices = GT_indices.union(sp)

        GT = G_total.induced_subgraph(GT_indices)
        GTs.append(GT)
    
    return(GTs, GT_abstracts)


def clusterpairs_by_distance(G, G_total, clusters, clusterinfo, return_distances = False, verbose = False, full_run = False):
    """Calculates the (weighted) graph distances on G for a number of clusters.
    Returns all pairs of cluster ids and closest nodes in ascending order of their distance. 
    If return_distances, then distances are also returned.

    Returns a list containing these elements, sorted by distance:
    [(clusterid1, clusterid2), (closestnodeid1, closestnodeid2), distance]
    """
    
    cluster_indices = clusterindices_by_length(clusterinfo, False) # Start with the smallest so the for loop is as short as possible
    clusterpairs = []
    clustercopies = {}
    
    # Create copies of all clusters
    for i in range(len(cluster_indices)):
        clustercopies[i] = clusters[i].copy()
        
    # Take one cluster
    for i, c1 in enumerate(cluster_indices[:-1]):
        c1_indices = G_total.vs.select(lambda x: x["id"] in clustercopies[c1].vs()["id"]).indices
        print("Working on cluster " + str(i+1) + " of " + str(len(cluster_indices)) + "...")
        for j, c2 in enumerate(cluster_indices[i+1:]):
            closest_pair = {'i': -1, 'j': -1}
            min_dist = np.inf
            c2_indices = G_total.vs.select(lambda x: x["id"] in clustercopies[c2].vs()["id"]).indices
            if verbose: print("... routing " + str(len(c1_indices)) + " nodes to " + str(len(c2_indices)) + " nodes in other cluster " + str(j+1) + " of " + str(len(cluster_indices[i+1:])) + ".")
            
            if full_run:
                # Compare all pairs of nodes in both clusters (takes long)
                for a in list(c1_indices):
                    sp = G_total.get_shortest_paths(a, c2_indices, weights = "weight", output = "epath")

                    if all([not elem for elem in sp]):
                        # If there is no path from one node, there is no path from any node
                        break
                    else:
                        for path, c2_index in zip(sp, c2_indices):
                            if len(path) >= 1:
                                dist_nodes = sum([G_total.es[e]['weight'] for e in path])
                                if dist_nodes < min_dist:
                                    closest_pair['i'] = G_total.vs[a]["id"]
                                    closest_pair['j'] = G_total.vs[c2_index]["id"]
                                    min_dist = dist_nodes
            else:
                # Do a heuristic that should be close enough.
                # From cluster 1, look at all shortest paths only from its centroid
                a = clusterinfo[c1]["centroid_index"]
                sp = G_total.get_shortest_paths(a, c2_indices, weights = "weight", output = "epath")
                if all([not elem for elem in sp]):
                    # If there is no path from one node, there is no path from any node
                    break
                else:
                    for path, c2_index in zip(sp, c2_indices):
                        if len(path) >= 1:
                            dist_nodes = sum([G_total.es[e]['weight'] for e in path])
                            if dist_nodes < min_dist:
                                closest_pair['j'] = G_total.vs[c2_index]["id"]
                                min_dist = dist_nodes
                # Closest c2 node to centroid1 found. Now find all c1 nodes to that closest c2 node.
                b = G_total.vs.find(id = closest_pair['j']).index
                sp = G_total.get_shortest_paths(b, c1_indices, weights = "weight", output = "epath")
                if all([not elem for elem in sp]):
                    # If there is no path from one node, there is no path from any node
                    break
                else:
                    for path, c1_index in zip(sp, c1_indices):
                        if len(path) >= 1:
                            dist_nodes = sum([G_total.es[e]['weight'] for e in path])
                            if dist_nodes <= min_dist: # <=, not <!
                                closest_pair['i'] = G_total.vs[c1_index]["id"]
                                min_dist = dist_nodes
            
            if closest_pair['i'] != -1 and closest_pair['j'] != -1:
                clusterpairs.append([(c1, c2), (closest_pair['i'], closest_pair['j']), min_dist])
                                    
    clusterpairs.sort(key = lambda x: x[-1])
    if return_distances:
        return clusterpairs
    else:
        return [[o[0], o[1]] for o in clusterpairs]


def mst_routing(G, pois):
    """Minimum Spanning Tree (MST) of a graph G's node subset pois,
    then routing to connect the MST.
    G is an ipgraph graph, pois is a list of node ids.
    
    The MST is the planar graph with the minimum number of (weighted) 
    links in order to assure connectedness.

    Distance here is routing distance, while edge crossing is checked on an abstract 
    level.
    """

    if len(pois) < 2: return (ig.Graph(), ig.Graph()) # We can't do anything with less than 2 POIs

    # MST_abstract is the MST with same nodes but euclidian links
    pois_indices = set()
    for poi in pois:
        pois_indices.add(G.vs.find(id = poi).index)
    G_temp = copy.deepcopy(G)
    for e in G_temp.es: # delete all edges
        G_temp.es.delete(e)
        
    poipairs = poipairs_by_distance(G, pois, True)
    if len(poipairs) == 0: return (ig.Graph(), ig.Graph())

    MST_abstract = copy.deepcopy(G_temp.subgraph(pois_indices))
    for poipair, poipair_distance in poipairs:
        poipair_ind = (MST_abstract.vs.find(id = poipair[0]).index, MST_abstract.vs.find(id = poipair[1]).index)
        MST_abstract.add_edge(poipair_ind[0], poipair_ind[1] , weight = poipair_distance)
    MST_abstract = MST_abstract.spanning_tree(weights = "weight")

    # Get node pairs we need to route, sorted by distance
    routenodepairs = {}
    for e in MST_abstract.es:
        routenodepairs[(e.source_vertex["id"], e.target_vertex["id"])] = e["weight"]
    routenodepairs = sorted(routenodepairs.items(), key = lambda x: x[1])

    # Do the routing
    MST_indices = set()
    for poipair, poipair_distance in routenodepairs:
        poipair_ind = (G.vs.find(id = poipair[0]).index, G.vs.find(id = poipair[1]).index)
        sp = set(G.get_shortest_paths(poipair_ind[0], poipair_ind[1], weights = "weight", output = "vpath")[0])
        MST_indices = MST_indices.union(sp)

    MST = G.induced_subgraph(MST_indices)
    
    return (MST, MST_abstract)



def greedy_triangulation(GT, poipairs, prune_quantile = 1, prune_measure = "betweenness", edgeorder = False):
    """Greedy Triangulation (GT) of a graph GT with an empty edge set.
    Distances between pairs of nodes are given by poipairs.
    
    The GT connects pairs of nodes in ascending order of their distance provided
    that no edge crossing is introduced. It leads to a maximal connected planar
    graph, while minimizing the total length of edges considered. 
    See: cardillo2006spp
    """
    
    for poipair, poipair_distance in poipairs:
        poipair_ind = (GT.vs.find(id = poipair[0]).index, GT.vs.find(id = poipair[1]).index)
        if not new_edge_intersects(GT, (GT.vs[poipair_ind[0]]["x"], GT.vs[poipair_ind[0]]["y"], GT.vs[poipair_ind[1]]["x"], GT.vs[poipair_ind[1]]["y"])):
            GT.add_edge(poipair_ind[0], poipair_ind[1], weight = poipair_distance)
            
    # Get the measure for pruning
    if prune_measure == "betweenness":
        BW = GT.edge_betweenness(directed = False, weights = "weight")
        qt = np.quantile(BW, 1-prune_quantile)
        sub_edges = []
        for c, e in enumerate(GT.es):
            if BW[c] >= qt: 
                sub_edges.append(c)
            GT.es[c]["bw"] = BW[c]
            GT.es[c]["width"] = math.sqrt(BW[c]+1)*0.5
        # Prune
        GT = GT.subgraph_edges(sub_edges)
    elif prune_measure == "closeness":
        CC = GT.closeness(vertices = None, weights = "weight")
        qt = np.quantile(CC, 1-prune_quantile)
        sub_nodes = []
        for c, v in enumerate(GT.vs):
            if CC[c] >= qt: 
                sub_nodes.append(c)
            GT.vs[c]["cc"] = CC[c]
        GT = GT.induced_subgraph(sub_nodes)
    elif prune_measure == "random":
        ind = np.quantile(np.arange(len(edgeorder)), prune_quantile, interpolation = "lower") + 1 # "lower" and + 1 so smallest quantile has at least one edge
        GT = GT.subgraph_edges(edgeorder[:ind])
    
    return GT


def greedy_triangulation_routing(G, pois, prune_quantiles = [1], prune_measure = "betweenness"):
    """Greedy Triangulation (GT) of a graph G's node subset pois,
    then routing to connect the GT (up to a quantile of betweenness
    betweenness_quantile).
    G is an ipgraph graph, pois is a list of node ids.
    
    The GT connects pairs of nodes in ascending order of their distance provided
    that no edge crossing is introduced. It leads to a maximal connected planar
    graph, while minimizing the total length of edges considered. 
    See: cardillo2006spp
    
    Distance here is routing distance, while edge crossing is checked on an abstract 
    level.
    """
    
    if len(pois) < 2: return ([], []) # We can't do anything with less than 2 POIs

    # GT_abstract is the GT with same nodes but euclidian links to keep track of edge crossings
    pois_indices = set()
    for poi in pois:
        pois_indices.add(G.vs.find(id = poi).index)
    G_temp = copy.deepcopy(G)
    for e in G_temp.es: # delete all edges
        G_temp.es.delete(e)
        
    poipairs = poipairs_by_distance(G, pois, True)
    if len(poipairs) == 0: return ([], [])

    if prune_measure == "random":
        # run the whole GT first
        GT = copy.deepcopy(G_temp.subgraph(pois_indices))
        for poipair, poipair_distance in poipairs:
            poipair_ind = (GT.vs.find(id = poipair[0]).index, GT.vs.find(id = poipair[1]).index)
            if not new_edge_intersects(GT, (GT.vs[poipair_ind[0]]["x"], GT.vs[poipair_ind[0]]["y"], GT.vs[poipair_ind[1]]["x"], GT.vs[poipair_ind[1]]["y"])):
                GT.add_edge(poipair_ind[0], poipair_ind[1], weight = poipair_distance)
        # create a random order for the edges
        random.seed(0) # const seed for reproducibility
        edgeorder = random.sample(range(GT.ecount()), k = GT.ecount())
    else: 
        edgeorder = False
    
    GT_abstracts = []
    GTs = []
    for prune_quantile in tqdm(prune_quantiles, desc = "Greedy triangulation", leave = False):
        GT_abstract = copy.deepcopy(G_temp.subgraph(pois_indices))
        GT_abstract = greedy_triangulation(GT_abstract, poipairs, prune_quantile, prune_measure, edgeorder)
        GT_abstracts.append(GT_abstract)
        
        # Get node pairs we need to route, sorted by distance
        routenodepairs = {}
        for e in GT_abstract.es:
            routenodepairs[(e.source_vertex["id"], e.target_vertex["id"])] = e["weight"]
        routenodepairs = sorted(routenodepairs.items(), key = lambda x: x[1])

        # Do the routing
        GT_indices = set()
        for poipair, poipair_distance in routenodepairs:
            poipair_ind = (G.vs.find(id = poipair[0]).index, G.vs.find(id = poipair[1]).index)
            sp = set(G.get_shortest_paths(poipair_ind[0], poipair_ind[1], weights = "weight", output = "vpath")[0])
            GT_indices = GT_indices.union(sp)

        GT = G.induced_subgraph(GT_indices)
        GTs.append(GT)
    
    return (GTs, GT_abstracts)
    
    
def poipairs_by_distance(G, pois, return_distances = False):
    """Calculates the (weighted) graph distances on G for a subset of nodes pois.
    Returns all pairs of poi ids in ascending order of their distance. 
    If return_distances, then distances are also returned.
    """
    
    # Get poi indices
    indices = []
    for poi in pois:
        indices.append(G_carall.vs.find(id = poi).index)
    
    # Get sequences of nodes and edges in shortest paths between all pairs of pois
    poi_nodes = []
    poi_edges = []
    for c, v in enumerate(indices):
        poi_nodes.append(G.get_shortest_paths(v, indices[c:], weights = "weight", output = "vpath"))
        poi_edges.append(G.get_shortest_paths(v, indices[c:], weights = "weight", output = "epath"))

    # Sum up weights (distances) of all paths
    poi_dist = {}
    for paths_n, paths_e in zip(poi_nodes, poi_edges):
        for path_n, path_e in zip(paths_n, paths_e):
            # Sum up distances of path segments from first to last node
            path_dist = sum([G.es[e]['weight'] for e in path_e])
            if path_dist > 0:
                poi_dist[(path_n[0],path_n[-1])] = path_dist
            
    temp = sorted(poi_dist.items(), key = lambda x: x[1])
    # Back to ids
    output = []
    for p in temp:
        output.append([(G.vs[p[0][0]]["id"], G.vs[p[0][1]]["id"]), p[1]])
    
    if return_distances:
        return output
    else:
        return [o[0] for o in output]





# ANALYSIS

def rotate_grid(p, origin = (0, 0), degrees = 0):
        """Rotate a list of points around an origin (in 2D). 
        
        Parameters:
            p (tuple or list of tuples): (x,y) coordinates of points to rotate
            origin (tuple): (x,y) coordinates of rotation origin
            degrees (int or float): degree (clockwise)

        Returns:
            ndarray: the rotated points, as an ndarray of 1x2 ndarrays
        """
        # https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
        angle = np.deg2rad(-degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T-o.T) + o.T).T)


# Two functions from: https://github.com/gboeing/osmnx-examples/blob/v0.11/notebooks/17-street-network-orientations.ipynb
def reverse_bearing(x):
    return x + 180 if x < 180 else x - 180

def count_and_merge(n, bearings):
    # make twice as many bins as desired, then merge them in pairs
    # prevents bin-edge effects around common values like 0° and 90°
    n = n * 2
    bins = np.arange(n + 1) * 360 / n
    count, _ = np.histogram(bearings, bins=bins)
    
    # move the last bin to the front, so eg 0.01° and 359.99° will be binned together
    count = np.roll(count, 1)
    return count[::2] + count[1::2]


def calculate_directness(G, numnodepairs = 500):
    """Calculate directness on G over all connected node pairs in indices.
    """
    
    indices = random.sample(list(G.vs), min(numnodepairs, len(G.vs)))

    poi_edges = []
    v1 = []
    v2 = []
    total_distance_haversine = 0
    for c, v in enumerate(indices):
        poi_edges.append(G.get_shortest_paths(v, indices[c:], weights = "weight", output = "epath"))
        temp = G.get_shortest_paths(v, indices[c:], weights = "weight", output = "vpath")
        total_distance_haversine += sum(haversine_vector([(G.vs[t[0]]["x"], G.vs[t[0]]["y"]) for t in temp], [(G.vs[t[-1]]["x"], G.vs[t[-1]]["y"]) for t in temp]))
    
    total_distance_network = 0
    for paths_e in poi_edges:
        for path_e in paths_e:
            # Sum up distances of path segments from first to last node
            total_distance_network += sum([G.es[e]['weight'] for e in path_e])
    
    return total_distance_haversine / total_distance_network


def listmean(lst): 
    try: return sum(lst) / len(lst)
    except: return 0

def calculate_coverage_edges(G, buffer_km = 0.5, return_cov = False, G_prev = ig.Graph(), cov_prev = Polygon()):
    """Calculates the area and shape covered by the graph's edges.
    If G_prev and cov_prev are given, only the difference between G and G_prev are calculated, then added to cov_prev.
    """

    G_added = copy.deepcopy(G)
    delete_overlaps(G_added, G_prev)

    # https://gis.stackexchange.com/questions/121256/creating-a-circle-with-radius-in-metres
    loncenter = listmean([v["x"] for v in G.vs])
    latcenter = listmean([v["y"] for v in G.vs])
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(latcenter, loncenter)
    # Use transformer: https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    wgs84_to_aeqd = pyproj.Transformer.from_proj(
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))
    aeqd_to_wgs84 = pyproj.Transformer.from_proj(
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"))
    edgetuples = [((e.source_vertex["x"], e.source_vertex["y"]), (e.target_vertex["x"], e.target_vertex["y"])) for e in G_added.es]
    # Shapely buffer seems slow for complex objects: https://stackoverflow.com/questions/57753813/speed-up-shapely-buffer
    # Therefore we buffer piecewise.
    cov_added = Polygon()
    for c, t in enumerate(edgetuples):
        # if cov.geom_type == 'MultiPolygon' and c % 1000 == 0: print(str(c)+"/"+str(len(edgetuples)), sum([len(pol.exterior.coords) for pol in cov]))
        # elif cov.geom_type == 'Polygon' and c % 1000 == 0: print(str(c)+"/"+str(len(edgetuples)), len(pol.exterior.coords))
        buf = ops.transform(aeqd_to_wgs84.transform, ops.transform(wgs84_to_aeqd.transform, LineString(t)).buffer(buffer_km * 1000))
        cov_added = ops.unary_union([cov_added, Polygon(buf)])

    # Merge with cov_prev
    if not cov_added.is_empty: # We need this check because apparently an empty Polygon adds an area.
        cov = ops.unary_union([cov_added, cov_prev])
    else:
        cov = cov_prev

    cov_transformed = ops.transform(wgs84_to_aeqd.transform, cov)
    covered_area = cov_transformed.area / 1000000

    if return_cov:
        return (covered_area, cov)
    else:
        return covered_area


def calculate_poiscovered(G, cov, nnids):
    """Calculates how many nodes, given by nnids, are covered by the shapely (multi)polygon cov
    """
    
    pois_indices = set()
    for poi in nnids:
        pois_indices.add(G.vs.find(id = poi).index)

    poiscovered = 0
    for poi in pois_indices:
        v = G.vs[poi]
        if Point(v["x"], v["y"]).within(cov):
            poiscovered += 1
    
    return poiscovered


def calculate_efficiency_global(G, numnodepairs = 500, normalized = True):
    """Calculates global network efficiency.
    If there are more than numnodepairs nodes, measure over pairings of a 
    random sample of numnodepairs nodes.
    """

    if G is None: return 0
    if G.vcount() > numnodepairs:
        nodeindices = random.sample(list(G.vs.indices), numnodepairs)
    else:
        nodeindices = list(G.vs.indices)
    d_ij = G.shortest_paths(source = nodeindices, target = nodeindices, weights = "weight")
    d_ij = [item for sublist in d_ij for item in sublist] # flatten
    EG = sum([1/d for d in d_ij if d != 0])
    if not normalized: return EG
    pairs = list(itertools.permutations(nodeindices, 2))
    if len(pairs) < 1: return 0
    l_ij = haversine_vector([(G.vs[p[0]]["x"], G.vs[p[0]]["y"]) for p in pairs],
                            [(G.vs[p[1]]["x"], G.vs[p[1]]["y"]) for p in pairs])
    EG_id = sum([1/l for l in l_ij if l != 0])
    # if (EG / EG_id) > 1: # This should not be allowed to happen!
    #     pp.pprint(d_ij)
    #     pp.pprint(l_ij)
    #     pp.pprint([e for e in G.es])
    #     print(pairs)
    #     print([(G.vs[p[0]]["x"], G.vs[p[0]]["y"]) for p in pairs],
    #                         [(G.vs[p[1]]["x"], G.vs[p[1]]["y"]) for p in pairs])
    #     print(EG, EG_id)
    #     sys.exit()
    # assert EG / EG_id <= 1, "Normalized EG > 1. This should not be possible."
    return EG / EG_id


def calculate_efficiency_local(G, numnodepairs = 500, normalized = True):
    """Calculates local network efficiency.
    If there are more than numnodepairs nodes, measure over pairings of a 
    random sample of numnodepairs nodes.
    """

    if G is None: return 0
    if G.vcount() > numnodepairs:
        nodeindices = random.sample(list(G.vs.indices), numnodepairs)
    else:
        nodeindices = list(G.vs.indices)
    EGi = []
    vcounts = []
    ecounts = []
    for i in nodeindices:
        if len(G.neighbors(i)) > 1: # If we have a nontrivial neighborhood
            G_induced = G.induced_subgraph(G.neighbors(i))
            EGi.append(calculate_efficiency_global(G_induced, numnodepairs, normalized))
    return listmean(EGi)


def calculate_metrics(G, GT_abstract, G_big, nnids, calcmetrics = {"length":0,
          "length_lcc":0,
          "coverage": 0,
          "directness": 0,
          "directness_lcc": 0,
          "poi_coverage": 0,
          "components": 0,
          "overlap_biketrack": 0,
          "overlap_bikeable": 0,
          "efficiency_global": 0,
          "efficiency_local": 0
         }, buffer_walk = 500, numnodepairs = 500, verbose = False, return_cov = True, G_prev = ig.Graph(), cov_prev = Polygon(), ignore_GT_abstract = False, Gexisting = {}):
    """Calculates all metrics (using the keys from calcmetrics).
    """
    
    output = {}
    for key in calcmetrics:
        output[key] = 0
    cov = Polygon()

    # Check that the graph has links (sometimes we have an isolated node)
    if G.ecount() > 0 and GT_abstract.ecount() > 0:

        # Get LCC
        cl = G.clusters()
        LCC = cl.giant()

        # EFFICIENCY
        if not ignore_GT_abstract:
            if verbose and ("efficiency_global" in calcmetrics or "efficiency_local" in calcmetrics): print("Calculating efficiency...")
            if "efficiency_global" in calcmetrics:
                output["efficiency_global"] = calculate_efficiency_global(GT_abstract, numnodepairs)
            if "efficiency_local" in calcmetrics:
                output["efficiency_local"] = calculate_efficiency_local(GT_abstract, numnodepairs) 
        
        # LENGTH
        if verbose and ("length" in calcmetrics or "length_lcc" in calcmetrics): print("Calculating length...")
        if "length" in calcmetrics:
            output["length"] = sum([e['weight'] for e in G.es])
        if "length_lcc" in calcmetrics:
            if len(cl) > 1:
                output["length_lcc"] = sum([e['weight'] for e in LCC.es])
            else:
                output["length_lcc"] = output["length"]
        
        # COVERAGE
        if "coverage" in calcmetrics:
            if verbose: print("Calculating coverage...")
            # G_added = G.difference(G_prev) # This doesnt work
            covered_area, cov = calculate_coverage_edges(G, buffer_walk/1000, return_cov, G_prev, cov_prev)
            output["coverage"] = covered_area
            # OVERLAP WITH EXISTING NETS
            if Gexisting:
                if "overlap_biketrack" in calcmetrics:
                    output["overlap_biketrack"] = edge_lengths(intersect_igraphs(Gexisting["biketrack"], G))
                if "overlap_bikeable" in calcmetrics:
                    output["overlap_bikeable"] = edge_lengths(intersect_igraphs(Gexisting["bikeable"], G))

        # POI COVERAGE
        if "poi_coverage" in calcmetrics:
            if verbose: print("Calculating POI coverage...")
            output["poi_coverage"] = calculate_poiscovered(G_big, cov, nnids)

        # COMPONENTS
        if "components" in calcmetrics:
            if verbose: print("Calculating components...")
            output["components"] = len(list(G.components()))
        
        # DIRECTNESS
        if verbose and ("directness" in calcmetrics or "directness_lcc" in calcmetrics): print("Calculating directness...")
        if "directness" in calcmetrics:
            output["directness"] = calculate_directness(G, numnodepairs)
        if "directness_lcc" in calcmetrics:
            if len(cl) > 1:
                output["directness_lcc"] = calculate_directness(LCC, numnodepairs)
            else:
                output["directness_lcc"] = output["directness"]

    if return_cov: 
        return (output, cov)
    else:
        return output


def overlap_linepoly(l, p):
    """Calculates the length of shapely LineString l falling inside the shapely Polygon p
    """
    return p.intersection(l).length if l.length else 0


def edge_lengths(G):
    """Returns the total length of edges in an igraph graph.
    """
    return sum([e['weight'] for e in G.es])


def intersect_igraphs(G1, G2):
    """Generates the graph intersection of igraph graphs G1 and G2, copying also link and node attributes.
    """
    # Ginter = G1.__and__(G2) # This does not work with attributes.
    if G1.ecount() > G2.ecount(): # Iterate through edges of the smaller graph
        G1, G2 = G2, G1
    inter_nodes = set()
    inter_edges = []
    inter_edge_attributes = {}
    inter_node_attributes = {}
    edge_attribute_name_list = G2.edge_attributes()
    node_attribute_name_list = G2.vertex_attributes()
    for edge_attribute_name in edge_attribute_name_list:
        inter_edge_attributes[edge_attribute_name] = []
    for node_attribute_name in node_attribute_name_list:
        inter_node_attributes[node_attribute_name] = []
    for e in list(G1.es):
        n1_id = e.source_vertex["id"]
        n2_id = e.target_vertex["id"]
        try:
            n1_index = G2.vs.find(id = n1_id).index
            n2_index = G2.vs.find(id = n2_id).index
        except ValueError:
            continue
        if G2.are_connected(n1_index, n2_index):
            inter_edges.append((n1_index, n2_index))
            inter_nodes.add(n1_index)
            inter_nodes.add(n2_index)
            edge_attributes = e.attributes()
            for edge_attribute_name in edge_attribute_name_list:
                inter_edge_attributes[edge_attribute_name].append(edge_attributes[edge_attribute_name])

    # map nodeids to first len(inter_nodes) integers
    idmap = {n_index:i for n_index,i in zip(inter_nodes, range(len(inter_nodes)))}

    G_inter = ig.Graph()
    G_inter.add_vertices(len(inter_nodes))
    G_inter.add_edges([(idmap[e[0]], idmap[e[1]]) for e in inter_edges])
    for edge_attribute_name in edge_attribute_name_list:
        G_inter.es[edge_attribute_name] = inter_edge_attributes[edge_attribute_name]

    for n_index in idmap.keys():
        v = G2.vs[n_index]
        node_attributes = v.attributes()
        for node_attribute_name in node_attribute_name_list:
            inter_node_attributes[node_attribute_name].append(node_attributes[node_attribute_name])
    for node_attribute_name in node_attribute_name_list:
        G_inter.vs[node_attribute_name] = inter_node_attributes[node_attribute_name]

    return G_inter


def calculate_metrics_additively(Gs, GT_abstracts, prune_quantiles, G_big, nnids, buffer_walk = 500, numnodepairs = 500, verbose = False, return_cov = True, Gexisting = {}):
    """Calculates all metrics, additively. 
    Coverage differences are calculated in every step instead of the whole coverage.
    """

    # BICYCLE NETWORKS
    output = {
            "length":[],
            "length_lcc":[],
            "coverage": [],
            "directness": [],
            "directness_lcc": [],
            "poi_coverage": [],
            "components": [],
            "overlap_biketrack": [],
            "overlap_bikeable": [],
            "efficiency_global": [],
            "efficiency_local": []
            }
    covs = {} # covers using buffer_walk
    cov_prev = Polygon()
    GT_prev = ig.Graph()
    for GT, GT_abstract, prune_quantile in zip(Gs, GT_abstracts, tqdm(prune_quantiles, desc = "Bicycle networks", leave = False)):
        if verbose: print("Calculating bike network metrics for quantile " + str(prune_quantile))
        metrics, cov = calculate_metrics(GT, GT_abstract, G_big, nnids, output, buffer_walk, numnodepairs, verbose, return_cov, GT_prev, cov_prev, False, Gexisting)
        
        for key in output.keys():
            output[key].append(metrics[key])
        covs[prune_quantile] = cov
        cov_prev = copy.deepcopy(cov)
        GT_prev = copy.deepcopy(GT)


    # # CAR CONSTRICTED BICYCLE NETWORKS (takes too long - commented out for now)
    # # These are the car networks where the length of the bike subnetwork is increased 10 times, effectively implementing a speed reduction from 50 km/h to 5 km/h. We are only interested in directness, as all other metrics do not change or are already calculated elsewhere.
    # output_carconstrictedbike = {
    #           "directness": [],
    #           "directness_lcc": []
    #          }
    # for GT, GT_abstract, prune_quantile in zip(Gs, GT_abstracts, tqdm(prune_quantiles, desc = "Car constricted bicycle networks", leave = False)):
    #     GT_carconstrictedbike = copy.deepcopy(G_big)
    #     constrict_overlaps(GT_carconstrictedbike, GT)
    #     if verbose: print("Calculating carconstrictedbike network metrics for quantile " + str(prune_quantile))
    #     metrics = calculate_metrics(GT_carconstrictedbike, GT_abstract, G_big, nnids, output_carconstrictedbike, buffer_walk, numnodepairs, verbose, False)
        
    #     for key in output_carconstrictedbike.keys():
    #         output_carconstrictedbike[key].append(metrics[key])


    # # CAR MINUS BICYCLE NETWORKS
    # # These are the car networks where the links from the bike subnetworks are completely removed. Here we follow a reverse order to build up the costly cover calculations additively.
    # # First construct the negative networks
    # GT_carminusbikes = []
    # for GT, prune_quantile in zip(reversed(Gs), reversed(prune_quantiles)):
    #     GT_carminusbike = copy.deepcopy(G_big)
    #     delete_overlaps(GT_carminusbike, GT)
    #     GT_carminusbikes.append(GT_carminusbike)
    #     # print((GT_carminusbike.ecount() + GT.ecount()), GT_carminusbike.ecount(), GT.ecount()) # sanity check

    # output_carminusbike = {
    #         "length":[],
    #         "length_lcc":[],
    #         "coverage": [],
    #         "directness": [],
    #         "directness_lcc": [],
    #         "poi_coverage": [],
    #         "components": []
    #         }
    # covs_carminusbike = {}
    # cov_prev = Polygon()
    # GT_prev = ig.Graph()
    # for GT, prune_quantile in zip(GT_carminusbikes, tqdm(reversed(prune_quantiles), desc = "Car minus bicycle networks", leave = False)):
    #     if verbose: print("Calculating carminusbike network metrics for quantile " + str(prune_quantile))
    #     metrics, cov = calculate_metrics(GT, GT, G_big, nnids, output_carminusbike, buffer_walk, numnodepairs, verbose, return_cov, GT_prev, cov_prev, True)
        
    #     for key in output_carminusbike.keys():
    #         output_carminusbike[key].insert(0, metrics[key]) # append to beginning due to reversed order
    #     covs_carminusbike[prune_quantile] = cov
    #     cov_prev = copy.deepcopy(cov)
    #     GT_prev = copy.deepcopy(GT)

    # return (output, covs, output_carminusbike, covs_carminusbike, output_carconstrictedbike)
    return (output, covs)


def generate_video(placeid, imgname, duplicatelastframe = 5, verbose = True):
    """Generate a video from a set of images using OpenCV
    """
    # Code adapted from: https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python#44948030
    
    images = [img for img in os.listdir(PATH["plots_networks"] + placeid + "/") if img.startswith(placeid + imgname)]
    images.sort()
    frame = cv2.imread(os.path.join(PATH["plots_networks"] + placeid + "/", images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(PATH["videos"] + placeid + "/" + placeid + imgname + '.avi', 0, 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(PATH["plots_networks"] + placeid + "/", image)))
    # Add the last frame duplicatelastframe more times:
    for i in range(0, duplicatelastframe):
        video.write(cv2.imread(os.path.join(PATH["plots_networks"] + placeid + "/", images[-1])))

    cv2.destroyAllWindows()
    video.release()
    if verbose:
        print("Video " + placeid + imgname + '.avi generated from ' + str(len(images)) + " frames.")



def write_result(res, mode, placeid, poi_source, prune_measure, suffix, dictnested = {}):
    """Write results (pickle or dict to csv)
    """
    if mode == "pickle":
        openmode = "wb"
    else:
        openmode = "w"

    if poi_source:
        filename = placeid + '_poi_' + poi_source + "_" + prune_measure + suffix
    else:
        filename = placeid + "_" + prune_measure + suffix

    with open(PATH["results"] + placeid + "/" + filename, openmode) as f:
        if mode == "pickle":
            pickle.dump(res, f)
        elif mode == "dict":
            w = csv.writer(f)
            w.writerow(res.keys())
            try: # dict with list values
                w.writerows(zip(*res.values()))
            except: # dict with single values
                w.writerow(res.values())
        elif mode == "dictnested":
            # https://stackoverflow.com/questions/29400631/python-writing-nested-dictionary-to-csv
            fields = ['network'] + list(dictnested.keys())
            w = csv.DictWriter(f, fields)
            w.writeheader()
            for key, val in sorted(res.items()):
                row = {'network': key}
                row.update(val)
                w.writerow(row)


def gdf_to_geojson(gdf, properties):
    """Turn a gdf file into a GeoJSON.
    The gdf must consist only of geometries of type Point.
    Adapted from: https://geoffboeing.com/2015/10/exporting-python-data-geojson/
    """
    geojson = {'type':'FeatureCollection', 'features':[]}
    for _, row in gdf.iterrows():
        feature = {'type':'Feature',
                   'properties':{},
                   'geometry':{'type':'Point',
                               'coordinates':[]}}
        feature['geometry']['coordinates'] = [row.geometry.x, row.geometry.y]
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)
    return geojson



def ig_to_shapely(G):
    """Turn an igraph graph G to a shapely LineString
    """
    edgetuples = [((e.source_vertex["x"], e.source_vertex["y"]), (e.target_vertex["x"], e.target_vertex["y"])) for e in G.es]
    G_shapely = LineString()
    for t in edgetuples:
        G_shapely = ops.unary_union([G_shapely, LineString(t)])
    return G_shapely



print("Loaded functions.\n")
