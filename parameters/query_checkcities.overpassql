/*
This is an example Overpass query.
Try it out by pressing the Run button above!
You can find more examples with the Load tool.
*/
way
["bicycle"~"designated"]["highway"!~"cycleway"]["highway"!~"path"]["highway"!~"footway"]["highway"!~"service"]
["highway"!~"pedestrian"]
["highway"!~"unclassified"]
["cycleway"!~"track"]
["cycleway:right"!~"track"]
["cycleway:left"!~"track"]
  ({{bbox}});
/*added by auto repair*/
(._;>;);
/*end of auto repair*/
out skel;
