warnings.filterwarnings('ignore')
rerun_existing = True

for placeid, placeinfo in cities.items():
    print(placeid + ": Analyzing existing infrastructure.")
    
    # output_place is one static file for the existing city. This can be compared to the generated infrastructure.
    # Make a check if this file was already generated - it only needs to be done once. If not, generate it:
    filename = placeid + "_existing.csv"
    if rerun_existing or not os.path.isfile(PATH["results"] + placeid + "/" + filename):
        empty_metrics = {
                         "length":0,
                         "length_lcc":0,
                         "coverage": 0,
                         "directness": 0,
                         "directness_lcc": 0,
                         "poi_coverage": 0,
                         "components": 0,
                         "efficiency_global": 0,
                         "efficiency_local": 0,
                         "efficiency_global_routed": 0,
                         "efficiency_local_routed": 0,
                         "directness_lcc_linkwise": 0,
                         "directness_all_linkwise": 0
                        }
        output_place = {}
        for networktype in networktypes:
            output_place[networktype] = copy.deepcopy(empty_metrics)

        # Analyze all networks     
        Gs = {}
        for networktype in networktypes:
            if networktype != "biketrack_onstreet" and networktype != "bikeable_offstreet":
                Gs[networktype] = csv_to_ig(PATH["data"] + placeid + "/", placeid, networktype)
                Gs[networktype + "_simplified"] = csv_to_ig(PATH["data"] + placeid + "/", placeid, networktype + "_simplified")
            elif networktype == "biketrack_onstreet":
                Gs[networktype] = intersect_igraphs(Gs["biketrack"], Gs["carall"])
                Gs[networktype + "_simplified"] = intersect_igraphs(Gs["biketrack_simplified"], Gs["carall_simplified"])
            elif networktype == "bikeable_offstreet":
                G_temp = copy.deepcopy(Gs["bikeable"])
                delete_overlaps(G_temp, Gs["carall"])
                Gs[networktype] = G_temp
                G_temp = copy.deepcopy(Gs["bikeable_simplified"])
                delete_overlaps(G_temp, Gs["carall_simplified"])
                Gs[networktype + "_simplified"] = G_temp
        
        with open(PATH["data"] + placeid + "/" + placeid + '_poi_' + poi_source + '_nnidscarall.csv') as f:
            nnids = [int(line.rstrip()) for line in f]

            
        covs = {}
        for networktype in tqdm(networktypes, desc = "Networks", leave = False):
            if debug: print(placeid + ": Analyzing results: " + networktype)
            metrics, cov = calculate_metrics(Gs[networktype], Gs[networktype + "_simplified"], Gs['carall'], nnids, empty_metrics, buffer_walk, numnodepairs, debug)
            for key, val in metrics.items():
                output_place[networktype][key] = val
            covs[networktype] = cov
        # Save the covers
        write_result(covs, "pickle", placeid, "", "", "existing_covers.pickle")
        
        # Write to CSV
        write_result(output_place, "dictnested", placeid, "", "", "existing.csv", empty_metrics)



for placeid, placeinfo in cities.items():
    print(placeid + ": Analyzing results")

    # Load networks
    G_carall = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'carall')
    Gexisting = {}
    for networktype in ["biketrack", "bikeable"]:
        Gexisting[networktype] = csv_to_ig(PATH["data"] + placeid + "/", placeid, networktype)
        
    
    # Load POIs
    with open(PATH["data"] + placeid + "/" + placeid + '_poi_' + poi_source + '_nnidscarall.csv') as f:
        nnids = [int(line.rstrip()) for line in f]
            
    # Load results
    filename = placeid + '_poi_' + poi_source + "_" + prune_measure
    resultfile = open(PATH["results"] + placeid + "/" + filename + ".pickle",'rb')
    res = pickle.load(resultfile)
    resultfile.close()
    if debug: pp.pprint(res)
         
    # Calculate
    # output contains lists for all the prune_quantile values of the corresponding results
    output, covs = calculate_metrics_additively(res["GTs"], res["GT_abstracts"], res["prune_quantiles"], G_carall, nnids, buffer_walk, numnodepairs, debug, True, Gexisting)
    output_MST, cov_MST = calculate_metrics(res["MST"], res["MST_abstract"], G_carall, nnids, output, buffer_walk, numnodepairs, debug, True, ig.Graph(), Polygon(), False, Gexisting)
        
    # Save the covers
    write_result(covs, "pickle", placeid, poi_source, prune_measure, "_covers.pickle")
#     write_result(covs_carminusbike, "pickle", placeid, poi_source, prune_measure, "_covers_carminusbike.pickle")
    write_result(cov_MST, "pickle", placeid, poi_source, prune_measure, "_cover_mst.pickle")
        
    # Write to CSV
    write_result(output, "dict", placeid, poi_source, prune_measure, ".csv")
#     write_result(output_carminusbike, "dict", placeid, poi_source, prune_measure, "_carminusbike.csv")
#     write_result(output_carconstrictedbike, "dict", placeid, poi_source, prune_measure, "_carconstrictedbike.csv")
    write_result(output_MST, "dict", placeid, poi_source, "", "mst.csv")
