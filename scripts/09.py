# This script is copied from code/09_supplements.ipynb
# It supplements and updates existing results with additional calculations.

warnings.filterwarnings('ignore')

for placeid, placeinfo in cities.items():
    print(placeid + ": Analyzing results")

    # Load networks
    G_carall = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'carall')

    # Load POIs
    with open(PATH["data"] + placeid + "/" + placeid + '_poi_' + poi_source + '_nnidscarall.csv') as f:
        nnids = [int(line.rstrip()) for line in f]

    # Load results
    filename = placeid + '_poi_' + poi_source + "_" + prune_measure
    resultfile = open(PATH["results"] + placeid + "/" + filename + ".pickle",'rb')
    res = pickle.load(resultfile)
    resultfile.close()

    # Calculate
    # output contains lists for all the prune_quantile values of the corresponding results
    output, covs = calculate_metrics_additively(res["GTs"], res["GT_abstracts"], res["prune_quantiles"], G_carall, nnids, buffer_walk = buffer_walk, numnodepairs = numnodepairs, verbose = False, return_cov = True, Gexisting = {}, output = {"directness_lcc_linkwise": [], "directness_all_linkwise": []})

    # Read old results
    filename = placeid + '_poi_' + poi_source + "_" + prune_measure + ".csv"
    results_old = np.genfromtxt(PATH["results"] + placeid + "/" + filename, delimiter=',', names = True)

    # Stitch the results together
    output_final = {}
    for fieldname in results_old.dtype.names:
        if fieldname != "directness_lcc_linkwise" and fieldname != "directness_all_linkwise":
            output_final[fieldname] = list(results_old[fieldname])
    for fieldname in list(output.keys()):
        output_final[fieldname] = output[fieldname]

    # Overwrite old stuff
    write_result(output_final, "dict", placeid, poi_source, prune_measure, ".csv")

    
    # Same for MST
    output_MST, cov_MST = calculate_metrics(res["MST"], res["MST_abstract"], G_carall, nnids, calcmetrics ={"directness_lcc_linkwise": 0, "directness_all_linkwise": 0}, buffer_walk = buffer_walk, numnodepairs = numnodepairs, verbose = debug, return_cov = True, G_prev = ig.Graph(), cov_prev = Polygon(), ignore_GT_abstract = False, Gexisting = {})
    
    # Read old results
    filename = placeid + '_poi_' + poi_source + "_mst.csv"
    results_MST_old = np.genfromtxt(PATH["results"] + placeid + "/" + filename, delimiter=',', names = True)

    # Stitch the results together
    output_MST_final = {}
    for fieldname in results_MST_old.dtype.names:
        if fieldname != "directness_lcc_linkwise" and fieldname != "directness_all_linkwise":
            output_MST_final[fieldname] = results_MST_old[fieldname]
    for fieldname in list(output_MST.keys()):
        output_MST_final[fieldname] = output_MST[fieldname]

    # Overwrite old stuff
    write_result(output_MST_final, "dict", placeid, poi_source, "", "mst.csv")