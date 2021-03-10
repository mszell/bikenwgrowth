for placeid, placeinfo in cities.items():
    print(placeid + ": Exporting carconstrictedbike to picklez")
    
    # Load existing
    G_carall = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'carall')
    with open(PATH["exports"] + placeid + "/" + placeid + '_carall.picklez', 'wb') as f:
        G_carall_simplified = simplify_ig(G_carall)
        G_carall_simplified.write_picklez(fname = f)
    if debug: map_center = nxdraw(G_carall, "carall")
            
    # Load results
    filename = placeid + '_poi_' + poi_source + "_" + prune_measure
    resultfile = open(PATH["results"] + placeid + "/" + filename + ".pickle",'rb')
    res = pickle.load(resultfile)
    resultfile.close()
    
    if debug:
        fig = initplot()
        nxdraw(G_carall_simplified, "abstract", map_center, nodesize = 0, weighted = True, maxwidthsquared = 500)
        plt.savefig(PATH["exports"] + placeid + "/" + placeid + '_carallweighted.png', bbox_inches="tight", dpi=plotparam["dpi"])
        plt.close()
    for GT, prune_quantile in zip(res["GTs"], res["prune_quantiles"]):
        if prune_quantile in prune_quantiles:
            GT_carconstrictedbike = copy.deepcopy(G_carall)
            constrict_overlaps(GT_carconstrictedbike, GT)
            GT_carconstrictedbike = simplify_ig(GT_carconstrictedbike)
            if debug:
                fig = initplot()
                nxdraw(GT_carconstrictedbike, "abstract", map_center, nodesize = 0, weighted = True, maxwidthsquared = 500)
                plt.savefig(PATH["exports"] + placeid + "/" + placeid + '_carconstrictedbike_poi_' + poi_source + "_" + prune_measures[prune_measure] + "{:.3f}".format(prune_quantile) + '.png', bbox_inches="tight", dpi=plotparam["dpi"])
                plt.close()
            with open(PATH["exports"] + placeid + "/" + placeid + '_carconstrictedbike_poi_' + poi_source + "_" + prune_measures[prune_measure] + "{:.3f}".format(prune_quantile) + '.picklez', 'wb') as f:
                GT_carconstrictedbike.write_picklez(fname = f)