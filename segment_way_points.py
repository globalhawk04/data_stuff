## hi I am the start of the end of this project
## I will be the file that is cleaned and ready to use
## Good job getting this far

import json
import requests
import urllib
import re
import googlemaps
from geopy.distance import distance
from geopy.distance import geodesic
import math, numpy as np
from geographiclib.geodesic import Geodesic
from math import asin, atan2, cos, degrees, radians, sin
import time
import csv
from transformers import pipeline
import requests
from PIL import Image, ImageDraw
from transformers import AutoModelForObjectDetection
from transformers import AutoImageProcessor
import torch
import os
import json
import math 
import time
import decimal
import pandas as pd
import datetime




api_key = ''

filename = 'cleaned_google_waypoints_demo_1.csv'

points = []
predicted_lat = []
predicted_long = []
way_point_lat = []
way_point_long = []



with open(filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    next(csvfile)
    for row in datareader:
        lat = eval(row[1])
        longs = eval(row[2])
        points.append([lat,longs])   

print(len(points))          
drive_predicted_location = []
bearings = []

# Compute the route

for segments in range(len(points)-1):

    segment_1 = points[segments]
    segment_2 = points[segments+1]    
    latstart = segment_1[0]
    longstart = segment_1[1]
    latend = segment_2[0]
    longend = segment_2[1]
    result = Geodesic.WGS84.Inverse(latstart,longstart,latend,longend)
    distance = result["s12"] # in [m] (meters)
    bearing = result["azi1"] # in [°] (degrees)
   
    segmenting = round((distance/10)-1)
    if segmenting < 100:
        for i_like_to_move_it_move_it in range(segmenting):
            #print(i_like_to_move_it_move_it)
            move_it = i_like_to_move_it_move_it * .01
            bearings.append(bearing)
            R=6371
            lat1 = radians(latstart)
            lon1 = radians(longstart)
            a = radians(bearing)
            lat2 = asin(sin(lat1) * cos(move_it/R) + cos(lat1) * sin(move_it/R) * cos(a))
            lon2 = lon1 + atan2(
                sin(a) * sin(move_it/R) * cos(lat1),
                cos(move_it/R) - sin(lat1) * sin(lat2)
            )
            lat_end = degrees(lat2)
            long_end = degrees(lon2)
            #print(lat_end, long_end)
            location = str(lat_end)+','+str(long_end)
            drive_predicted_location.append(location)
            way_point_lat.append(lat_end)
            way_point_long.append(long_end)
            

print(len(drive_predicted_location))

df = pd.DataFrame({'predicted_lat':way_point_lat, 'predicted_long':way_point_long})
#df1 = df.drop_duplicates()
df.to_csv('demo_1_cleaned_way_point_test_florid.csv')
                    



