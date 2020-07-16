# System
import copy
import csv
import sys
import os
import watermark
import pickle
import itertools
import random
import pprint
pp = pprint.PrettyPrinter(indent=4)

# Math/Data
import math
import numpy as np
import pandas as pd

# Network
import igraph as ig
import networkx as nx

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm

# Geo
import osmnx as ox
# ox.utils.config(timeout = 300) # should work, but doc inconsistent: https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.utils.config
import fiona
import shapely
import gdal
import osr
from haversine import haversine, haversine_vector
import pyproj
from shapely.geometry import Point, MultiPoint, LineString, Polygon, MultiLineString
import shapely.ops as ops
import geopandas as gpd
from functools import partial


prune_quantiles = [x/20 for x in list(range(1, 21))] # The quantiles where the GT should be pruned using the prune_measure
prune_measures = {"betweenness": "Bq", "closeness": "Cq"}

osmnxparameters = {'car30': {'network_type':'drive', 'custom_filter':'["maxspeed"~"30"]', 'export': True},
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


# dict of placeid:placeinfo
# If a city has a proper shapefile through nominatim
# In case no (False), manual download of shapefile is necessary, see below
cities = {}
with open(PATH["parameters"] + 'cities.csv') as f:
    csvreader = csv.DictReader(f, delimiter=';')
    for row in csvreader:
        cities[row['placeid']] = {}
        for field in csvreader.fieldnames[1:]:
            cities[row['placeid']][field] = row[field]
            
if debug:
    print("\n\n=== Cities ===")
    pp.pprint(cities)
    print("==============\n\n")

# Create folders
for placeid, placeinfo in cities.items():
    placepath = PATH["data"] + placeid + "/"
    if not os.path.exists(placepath):
        os.makedirs(placepath)
        print("Successfully created folder " + placepath)


print("Setup finished\n")