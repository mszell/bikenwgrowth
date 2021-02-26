warnings.filterwarnings('ignore')

# %%capture
for placeid, placeinfo in cities.items():
    print(placeid + ": Plotting networks")
    
    if prune_measure == "betweenness":
        weight_abstract = True
    else:
        weight_abstract = 6
    
    # EXISTING INFRASTRUCTURE
    # Load networks
    G_biketrack = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'biketrack')
    G_carall = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'carall')
    G_biketrackcarall = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'biketrackcarall')
    G_bikeable = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'bikeable')
    map_center = nxdraw(G_carall, "carall")
    
    # PLOT existing networks
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_carall.pdf', bbox_inches="tight")
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_carall.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
    
    fig = initplot()
    nxdraw(G_biketrack, "biketrack", map_center)
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_biketrack.pdf', bbox_inches="tight")
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_biketrack.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
    
    fig = initplot()
    nxdraw(G_bikeable, "bikeable", map_center)
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_bikeable.pdf', bbox_inches="tight")
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_bikeable.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
    
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    nxdraw(G_biketrack, "biketrack", map_center, list(set([v["id"] for v in G_biketrack.vs]).intersection(set([v["id"] for v in G_carall.vs]))))
    nxdraw(G_biketrack, "biketrack_offstreet", map_center, list(set([v["id"] for v in G_biketrack.vs]).difference(set([v["id"] for v in G_carall.vs]))))
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_biketrackcarall.pdf', bbox_inches="tight")
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_biketrackcarall.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
    
    # Load POIs
    with open(PATH["data"] + placeid + "/" + placeid + '_poi_' + poi_source + '_nnidscarall.csv') as f:
        nnids = [int(line.rstrip()) for line in f]
    nodesize_poi = nodesize_from_pois(nnids)
    
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_carall_poi_' + poi_source + '.pdf', bbox_inches="tight")
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_carall_poi_' + poi_source + '.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()




# %%capture
for placeid, placeinfo in cities.items():
    print(placeid + ": Plotting networks")
    
    if prune_measure == "betweenness":
        weight_abstract = True
    else:
        weight_abstract = 6
    
    # EXISTING INFRASTRUCTURE
    # Load networks
    G_carall = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'carall')
    map_center = nxdraw(G_carall, "carall")
    
    # Load POIs
    with open(PATH["data"] + placeid + "/" + placeid + '_poi_' + poi_source + '_nnidscarall.csv') as f:
        nnids = [int(line.rstrip()) for line in f]
    nodesize_poi = nodesize_from_pois(nnids)
    
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_carall_poi_' + poi_source + '.pdf', bbox_inches="tight")
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_carall_poi_' + poi_source + '.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
    
    
    # GENERATED, POI BASED
    # Load results
    filename = placeid + '_poi_' + poi_source + "_" + prune_measure + ".pickle"
    with open(PATH["results"] + placeid + "/" + filename, 'rb') as f:
        res = pickle.load(f)
    if debug: pp.pprint(res)
        
    # PLOT abstract MST
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    nxdraw(res["MST_abstract"], "abstract", map_center, weighted = 6)
    nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
    nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in res["MST"].vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_MSTabstract_poi_' + poi_source + '.pdf', bbox_inches="tight")
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_MSTabstract_poi_' + poi_source + '.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
    
    # PLOT MST all together
    fig = initplot()
    nxdraw(G_carall, "carall")
    nxdraw(res["MST"], "bikegrown", map_center, nodesize = nodesize_grown)
    nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
    nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in res["MST"].vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_MSTall_poi_' + poi_source + '.pdf', bbox_inches="tight")
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_MSTall_poi_' + poi_source + '.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
    
    # PLOT MST all together with abstract
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    nxdraw(res["MST"], "bikegrown", map_center, nodesize = 0)
    nxdraw(res["MST_abstract"], "abstract", map_center, weighted = 6)
    nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
    nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in res["MST"].vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_MSTabstractall_poi_' + poi_source + '.pdf', bbox_inches="tight")
    plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_MSTabstractall_poi_' + poi_source + '.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
    
    # PLOT abstract greedy triangulation (this can take some minutes)
    for GT_abstract, prune_quantile in zip(res["GT_abstracts"], res["prune_quantiles"]):
        fig = initplot()
        nxdraw(G_carall, "carall")
        try:
            GT_abstract.es["weight"] = GT_abstract.es["width"]
        except:
            pass
        nxdraw(GT_abstract, "abstract", map_center, drawfunc = "nx.draw_networkx_edges", nodesize = 0, weighted = weight_abstract, maxwidthsquared = nodesize_poi)
        nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
        nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in GT_abstract.vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
        plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_GTabstract_poi_' + poi_source + "_" + prune_measures[prune_measure] + "{:.3f}".format(prune_quantile) + '.png', bbox_inches="tight", dpi=plotparam["dpi"])
        plt.close()
    
    # PLOT all together (this can take some minutes)
    for GT, prune_quantile in zip(res["GTs"], res["prune_quantiles"]):
        fig = initplot()
        nxdraw(G_carall, "carall")
        nxdraw(GT, "bikegrown", map_center, nodesize = nodesize_grown)
        nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
        nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in GT.vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
        plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_GTall_poi_' + poi_source + "_" + prune_measures[prune_measure] + "{:.3f}".format(prune_quantile) + '.png', bbox_inches="tight", dpi=plotparam["dpi"])
        plt.close()
        



# %%capture
for placeid, placeinfo in cities.items():
    print(placeid + ": Plotting network covers")

    # Load networks
    G_biketrack = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'biketrack')
    G_carall = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'carall')
    G_biketrackcarall = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'biketrackcarall')
    G_bikeable = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'bikeable')
    map_center = nxdraw(G_carall, "carall")
    
    # Load POIs
    with open(PATH["data"] + placeid + "/" + placeid + '_poi_' + poi_source + '_nnidscarall.csv') as f:
        nnids = [int(line.rstrip()) for line in f]
    nodesize_poi = nodesize_from_pois(nnids)
    
    # Load results
    filename = placeid + '_poi_' + poi_source + "_" + prune_measure + ".pickle"
    with open(PATH["results"] + placeid + "/" + filename, 'rb') as f:
        res = pickle.load(f)
    
    # Load covers
    filename = placeid + '_poi_' + poi_source + "_" + prune_measure + "_covers"
    with open(PATH["results"] + placeid + "/" + filename + ".pickle",'rb') as f:
        covs = pickle.load(f)
    filename = placeid + "_"  + "existing_covers"
    with open(PATH["results"] + placeid + "/" + filename + ".pickle",'rb') as f:
        cov_car = pickle.load(f)['carall']
    
    # Construct and plot patches from covers
    patchlist_car, patchlist_car_holes = cov_to_patchlist(cov_car, map_center)
    for GT, prune_quantile, cov in zip(res["GTs"], res["prune_quantiles"], covs.values()):
        fig = initplot()
        
        # Covers
        axes = fig.add_axes([0, 0, 1, 1]) # left, bottom, width, height (range 0 to 1)
        patchlist_bike, patchlist_bike_holes = cov_to_patchlist(cov, map_center)
        
        # We have this contrived order due to alphas, holes, and matplotlib's inability to draw polygon patches with holes. This only works because the car network is a superset of the bike network.
        # car orange, bike white, bike blue, bike holes white, bike holes orange, car holes white
        patchlist_combined = patchlist_car + patchlist_bike + patchlist_bike + patchlist_bike_holes+ patchlist_bike_holes + patchlist_car_holes
        pc = PatchCollection(patchlist_combined)
        colors = np.array([[255/255,115/255,56/255,0.2] for _ in range(len(patchlist_car))]) # car orange
        if len(patchlist_bike):
            colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_bike))], axis = 0) # bike white
            colors = np.append(colors, [[86/255,220/255,244/255,0.4] for _ in range(len(patchlist_bike))], axis = 0) # bike blue
        if len(patchlist_bike_holes):
            colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_bike_holes))], axis = 0) # bike holes white
        if len(patchlist_bike_holes):
            colors = np.append(colors, [[255/255,115/255,56/255,0.2] for _ in range(len(patchlist_bike_holes))], axis = 0) # bike holes orange
        if len(patchlist_car_holes):
            colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_car_holes))], axis = 0) # car holes white
        pc.set_facecolors(colors)
        pc.set_edgecolors(np.array([[0,0,0,0.4] for _ in range(len(patchlist_combined))])) # remove this line if the outline of the full coverage should remain
        axes.add_collection(pc)
        axes.set_aspect('equal')
        axes.set_xmargin(0.01)
        axes.set_ymargin(0.01)
        axes.plot()
        
        # Networks
        nxdraw(G_carall, "carall", map_center)
        nxdraw(GT, "bikegrown", map_center, nodesize = nodesize_grown)
        nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
        nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in GT.vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
        plt.savefig(PATH["plots_networks"] + placeid + "/" + placeid + '_GTallcover_poi_' + poi_source + "_" + prune_measures[prune_measure] + "{:.3f}".format(prune_quantile) + '.png', bbox_inches="tight", dpi=plotparam["dpi"])
        plt.close()
