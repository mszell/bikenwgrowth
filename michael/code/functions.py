
# GRAPH PLOTTING
def my_plot_reset(G, nids = False):
    reset_plot_attributes(G)
    color_nodes(G, "red", nids)
    size_nodes(G, 6, nids)

def reset_plot_attributes(G):
    """Resets node attributes for plotting.
    All black and size 0.
    """
    G.vs["color"] = "black"
    G.vs["size"] = 0
        
def color_nodes(G, color = "blue", nids = False, use_id = True):
    """Sets the color attribute of a set of nodes nids.
    """
    if nids is False:
        nids = [v.index for v in G.vs]
        use_id = False
    if use_id:
        for nid in set(nids):
            G.vs.find(id = nid)["color"] = color
    else:
        G.vs[nids]["color"] = color

def size_nodes(G, size = 6, nids = False, use_id = True):
    """Sets the size attribute of a set of nodes nids.
    """
    if nids is False:
        nids = [v.index for v in G.vs]
        use_id = False
    if use_id:
        for nid in set(nids):
            G.vs.find(id = nid)["size"] = size
    else:
        G.vs[nids]["size"] = size

def color_edges(G, color = "blue", eids = False):
    """Sets the color attribute of a set of edge eids.
    """
    if eids is False:
        G.es["color"] = color
    else:
        G.es[eids]["color"] = color
        
def width_edges(G, width = 1, eids = False):
    """Sets the width attribute of a set of edge eids.
    """
    if eids is False:
        G.es["width"] = width
    else:
        G.es[eids]["width"] = width
    

# OTHER FUNCTIONS
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

def ox_to_csv(G, p, placeid, parameterid, postfix = "", verbose = True):
    node,edge = ox.graph_to_gdfs(G)
    node.to_csv(p + placeid + '_' + parameterid + postfix + '_nodes.csv', index=False)
    edge.to_csv(p + placeid + '_' + parameterid + postfix + '_edges.csv', index=False)
    if verbose: print(placeid + ": Successfully wrote graph " + parameterid + postfix)

def csv_to_ig(p, placeid, parameterid):
    n = pd.read_csv(p + placeid + '_' + parameterid + '_nodes.csv')
    e = pd.read_csv(p + placeid + '_' + parameterid + '_edges.csv')
    G = osm_to_ig(n, e)
    round_coordinates(G)
    mirror_y(G)
    return(G)





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
    """Deletes all overlaps of G_res with G_orig (from G_res)
    based on node ids.
    """
    cnt_e = 0
    for e in list(G_res.es):
        try:
            n1_id = e.source_vertex["id"]
            n2_id = e.target_vertex["id"]
            # If there is already an edge in the original network, delete it
            n1_index = G_orig.vs.find(id = n1_id).index
            n2_index = G_orig.vs.find(id = n2_id).index
            if G_orig.are_connected(n1_index, n2_index):
                G_res.delete_edges(e)
                cnt_e += 1
        except:
            pass
    # Remove isolated nodes
    isolated_nodes = G_res.vs.select(_degree_eq=0)
    G_res.delete_vertices(isolated_nodes)
    if verbose: print("Removed " + str(cnt_e) + " overlapping edges and " + str(len(isolated_nodes)) + " nodes.")


def greedy_triangulation(GT, poipairs, prune_quantile = 1, prune_measure = "betweenness"):
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
    
    return GT
    

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
    
    GT_abstracts = []
    GTs = []
    for prune_quantile in prune_quantiles:
        GT_abstract = copy.deepcopy(G_temp.subgraph(pois_indices))
        GT_abstract = greedy_triangulation(GT_abstract, poipairs, prune_quantile, prune_measure)
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

def calculate_coverage_edges(G, buffer_km = 0.5, return_cov = False):
    """Calculates the area and shape covered by the graph's edges.
    """

    # https://gis.stackexchange.com/questions/121256/creating-a-circle-with-radius-in-metres
    latcenter = listmean([v["x"] for v in G.vs])
    loncenter = listmean([v["y"] for v in G.vs])
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(latcenter, loncenter)
    # Use transformer: https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    wgs84_to_aeqd = pyproj.Transformer.from_proj(
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection))
    aeqd_to_wgs84 = pyproj.Transformer.from_proj(
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"))
    edgetuples = [((e.source_vertex["x"], e.source_vertex["y"]), (e.target_vertex["x"], e.target_vertex["y"])) for e in G.es]
    # Shapely buffer seems slow for complex objects: https://stackoverflow.com/questions/57753813/speed-up-shapely-buffer
    # Therefore we buffer piecewise.
    cov = Polygon()
    for c, t in enumerate(edgetuples):
        # if cov.geom_type == 'MultiPolygon' and c % 1000 == 0: print(str(c)+"/"+str(len(edgetuples)), sum([len(pol.exterior.coords) for pol in cov]))
        # elif cov.geom_type == 'Polygon' and c % 1000 == 0: print(str(c)+"/"+str(len(edgetuples)), len(pol.exterior.coords))
        buf = ops.transform(aeqd_to_wgs84.transform, ops.transform(wgs84_to_aeqd.transform, LineString(t)).buffer(buffer_km * 1000))
        cov = ops.unary_union([cov, Polygon(buf)])
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
        if Point(-v["y"], v["x"]).within(cov):
            poiscovered += 1
    
    return poiscovered


def calculate_efficiency_global(G, numnodepairs = 500, normalized = True):
    """Calculates global network efficiency.
    """

    if len(list(G.vs)) > numnodepairs:
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
    return EG / EG_id

def calculate_efficiency_local(G, numnodepairs = 500, normalized = True):
    """Calculates local network efficiency.
    """
    if len(list(G.vs)) > numnodepairs*numnodepairs:
        nodeindices = random.sample(list(G.vs.indices), numnodepairs*numnodepairs)
    else:
        nodeindices = list(G.vs.indices)
    EGi = []
    for i in nodeindices:
        if len(G.neighbors(i)) > 1: # If we have a nontrivial neighborhood
            EGi.append(calculate_efficiency_global(G.induced_subgraph(G.neighbors(i)), normalized))
    return listmean(EGi)


def calculate_metrics(G, GT_abstract, G_big, nnids, buffer_walk = 500, numnodepairs = 500, verbose = False):
    """Calculates all metrics.
    """
    
    output = {"length":0,
          "coverage": 0,
          "directness": 0,
          "poi_coverage": 0,
          "components": 0,
          "efficiency_global": 0,
          "efficiency_local": 0
         }
    
    # EFFICIENCY
    if verbose: print("Calculating efficiency...")
    if GT_abstract is not None:
        output["efficiency_global"] = calculate_efficiency_global(GT_abstract, numnodepairs)
        output["efficiency_local"] = calculate_efficiency_local(GT_abstract, numnodepairs) 
    
    # LENGTH
    if verbose: print("Calculating length...")
    output["length"] = sum([e['weight'] for e in G.es])
    
    # COVERAGE
    if verbose: print("Calculating coverage...")
    covered_area, cov = calculate_coverage_edges(G, buffer_walk/1000, True)
    output["coverage"] = covered_area

    # POI COVERAGE
    if verbose: print("Calculating POI coverage...")
    output["poi_coverage"] = calculate_poiscovered(G_big, cov, nnids)

    # COMPONENTS
    if verbose: print("Calculating components...")
    output["components"] = len(list(G.components()))
    
    # DIRECTNESS
    if verbose: print("Calculating directness...")
    output["directness"] = calculate_directness(G, numnodepairs)

    return output



print("Loaded functions")