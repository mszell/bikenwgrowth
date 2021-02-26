for placeid, placeinfo in cities.items():
    print(placeid + ": Generating networks")

    # Load networks
    G_carall = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'carall')
    
    # Load POIs
    with open(PATH["data"] + placeid + "/" + placeid + '_poi_' + poi_source + '_nnidscarall.csv') as f:
        nnids = [int(line.rstrip()) for line in f]
    
    # Generation
    (GTs, GT_abstracts) = greedy_triangulation_routing(G_carall, nnids, prune_quantiles, prune_measure)
    (MST, MST_abstract) = mst_routing(G_carall, nnids)
    
    # Write results
    results = {"placeid": placeid, "prune_measure": prune_measure, "poi_source": poi_source, "prune_quantiles": prune_quantiles, "GTs": GTs, "GT_abstracts": GT_abstracts, "MST": MST, "MST_abstract": MST_abstract}
    write_result(results, "pickle", placeid, poi_source, prune_measure, ".pickle")
