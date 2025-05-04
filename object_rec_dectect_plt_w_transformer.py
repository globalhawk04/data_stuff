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
stamp = []
locations = [] 

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
df.to_csv('cleaned_way_point_test_florid.csv')
                    





for take_pic, direction in zip(drive_predicted_location,bearings):
    meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
    pic_base ='https://maps.googleapis.com/maps/api/streetview?'
    api_key = ''
    fov = '120'
    pitch = '10'
    heading = '90'
    met_params = {'key': api_key,
                  'location':take_pic,

                  }

    pic_params = {'key':api_key,
                  'location':take_pic,
                  'size':"640x640",
                  'heading':heading,
                  'fov':fov,
                  'pitch':pitch,
                  #its likely i will decrease the radius here because if i make the 
                  #adjustments i am making the latlongs segments should be more accurate
                  #and closer to the road.  
                  'radius':'20'
                  }

    meta_response = requests.get(meta_base, params=met_params)
    x = meta_response.json()
    try:
        lat_start = x['location']['lat']
        lng_start = x['location']['lng']
        meta_response = requests.get(meta_base, params=met_params)
    except Exception as e:
        pass

    #print(meta_response.json())
    time.sleep(1)
    
    pictures = str(lat_start)+(str(lng_start)+'jpg')
    
    pic_response = requests.get(pic_base, params=pic_params)

    with open('test.jpg', 'wb') as file:
        file.write(pic_response.content)
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        names = str(timestamp)
        # remember to close the response connection to the API
        pic_response.close()
        time.sleep(2)
        bests = 78
        score_check = [99.999999976]
        thinking = []
        
        image = Image.open('test.jpg')
      
        image_processor = AutoImageProcessor.from_pretrained("/home/j/Desktop/all_images/detr-5k_200e_1e-5_1e-4/"+str(bests))
        model = AutoModelForObjectDetection.from_pretrained("/home/j/Desktop/all_images/detr-5k_200e_1e-5_1e-4/"+str(bests))

        resulting = []
        boxes = []
        scores = []
        both = []
        image_name = []
        
        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            #here are the results they are going into a dictionary 
            results = image_processor.post_process_object_detection(outputs, threshold=0.00000001,
            target_sizes=target_sizes)[0]
            resulting.append(results)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]

            boxes.append(box)
            scores.append(score)
            boths = box , score.item()
            both.append(boths)
            
            
        filtered = sorted(both, key=lambda x:x[1])
        x = filtered[-1:]
        best_box = []
        if len(x) > 0:
            print('found a pole')
 
            print(x)
            print(lat_start,lng_start)
            for bbox in x:
            #print(bbox[1])
                score = 100.0000 - bbox[1]
                best_box.append(bbox[0])
                draw = ImageDraw.Draw(image)

                for box in best_box :
                    x, y, x2, y2 = tuple(box)
                    draw.rectangle((x, y, x2, y2), outline="red", width=1)
                    cropped_image = image.crop((x, y, x2, y2))
                    #this is cutting out and saving the cropped part of the image
                    #cropped_image.save(files)
                    google_camera_focal_length = 5.1
                    #distance = (focal_length * object_height) / image_height 
                    pic_height = 640
                    
                    object_width = cropped_image.size[0]
                    object_height = cropped_image.size[1]
                    #print(object_height)
                    
                    distance = (google_camera_focal_length * pic_height) / object_height
                    x_img= 640
                    y_img = 0
                    area = cropped_image.size[0] * cropped_image.size[1]
                    #print(area)

                    angle_from_camera = ((x - 320) *.28125) + int(heading)
                    print('found a pole.  it is this distance ' + '     '+str(distance)+'  from   ')
                    print(lat_start,lng_start) 
                    print('and at an angle of  '+ str(angle_from_camera))
                    print('  from the camera')
                    
                    R=6371
                    
                    distance = distance * .0002
                    
                    lat1 = radians(lat_start)
                    lon1 = radians(lng_start)
                    a = radians(angle_from_camera)
                    lat2 = asin(sin(lat1) * cos(distance/R) + cos(lat1) * sin(distance/R) * cos(a))
                    lon2 = lon1 + atan2(
                        sin(a) * sin(distance/R) * cos(lat1),
                        cos(distance/R) - sin(lat1) * sin(lat2)
                    )
                    lat_end = degrees(lat2)
                    long_end = degrees(lon2)
                    #print(lat_end, long_end)
                    location = str(lat_end)+','+str(long_end)
                    print('predicted location is   ')
                    print('*****************')
                    print(location)
                    stamp.append(names) 
                    image.save(names+'_location_'+str(location)+'.jpg')
                    predicted_lat.append(lat_end)
                    predicted_long.append(long_end)
                    locations.append(location) 
                    
                    


          
## hi I am the start of the end of this project
## I will be the file that is cleaned and ready to use
## Good job getting this far

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
df.to_csv('cleaned_way_point_test_florid.csv')
                    





for take_pic, direction in zip(drive_predicted_location,bearings):
    meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
    pic_base ='https://maps.googleapis.com/maps/api/streetview?'
    api_key = ''
    fov = '120'
    pitch = '10'
    heading = '270'
    met_params = {'key': api_key,
                  'location':take_pic,

                  }

    pic_params = {'key':api_key,
                  'location':take_pic,
                  'size':"640x640",
                  'heading':heading,
                  'fov':fov,
                  'pitch':pitch,
                  #its likely i will decrease the radius here because if i make the 
                  #adjustments i am making the latlongs segments should be more accurate
                  #and closer to the road.  
                  'radius':'20'
                  }

    meta_response = requests.get(meta_base, params=met_params)
    x = meta_response.json()
    try:
        lat_start = x['location']['lat']
        lng_start = x['location']['lng']
        meta_response = requests.get(meta_base, params=met_params)
    except Exception as e:
        pass

    #print(meta_response.json())
    time.sleep(1)
    
    pictures = str(lat_start)+(str(lng_start)+'jpg')
    
    pic_response = requests.get(pic_base, params=pic_params)

    with open('test.jpg', 'wb') as file:
        file.write(pic_response.content)
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        names = str(timestamp)
        # remember to close the response connection to the API
        pic_response.close()
        time.sleep(2)
        bests = 78
        score_check = [99.999999976]
        thinking = []
        
        image = Image.open('test.jpg')
      
        image_processor = AutoImageProcessor.from_pretrained("/home/j/Desktop/all_images/detr-5k_200e_1e-5_1e-4/"+str(bests))
        model = AutoModelForObjectDetection.from_pretrained("/home/j/Desktop/all_images/detr-5k_200e_1e-5_1e-4/"+str(bests))

        resulting = []
        boxes = []
        scores = []
        both = []
        image_name = []
        
        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            #here are the results they are going into a dictionary 
            results = image_processor.post_process_object_detection(outputs, threshold=0.00000001,
            target_sizes=target_sizes)[0]
            resulting.append(results)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]

            boxes.append(box)
            scores.append(score)
            boths = box , score.item()
            both.append(boths)
            
            
        filtered = sorted(both, key=lambda x:x[1])
        x = filtered[-1:]
        best_box = []
        if len(x) > 0:
            print('found a pole')
 
            print(x)
            print(lat_start,lng_start)
            for bbox in x:
            #print(bbox[1])
                score = 100.0000 - bbox[1]
                best_box.append(bbox[0])
                draw = ImageDraw.Draw(image)

                for box in best_box :
                    x, y, x2, y2 = tuple(box)
                    draw.rectangle((x, y, x2, y2), outline="red", width=1)
                    cropped_image = image.crop((x, y, x2, y2))
                    #this is cutting out and saving the cropped part of the image
                    #cropped_image.save(files)
                    google_camera_focal_length = 5.1
                    #distance = (focal_length * object_height) / image_height 
                    pic_height = 640
                    
                    object_width = cropped_image.size[0]
                    object_height = cropped_image.size[1]
                    #print(object_height)
                    
                    distance = (google_camera_focal_length * pic_height) / object_height
                    x_img= 640
                    y_img = 0
                    area = cropped_image.size[0] * cropped_image.size[1]
                    #print(area)

                    angle_from_camera = ((x - 320) *.28125) + int(heading)
                    print('found a pole.  it is this distance ' + '     '+str(distance)+'  from   ')
                    print(lat_start,lng_start) 
                    print('and at an angle of  '+ str(angle_from_camera))
                    print('  from the camera')
                    
                    R=6371
                    
                    distance = distance * .0002
                    
                    lat1 = radians(lat_start)
                    lon1 = radians(lng_start)
                    a = radians(angle_from_camera)
                    lat2 = asin(sin(lat1) * cos(distance/R) + cos(lat1) * sin(distance/R) * cos(a))
                    lon2 = lon1 + atan2(
                        sin(a) * sin(distance/R) * cos(lat1),
                        cos(distance/R) - sin(lat1) * sin(lat2)
                    )
                    lat_end = degrees(lat2)
                    long_end = degrees(lon2)
                    #print(lat_end, long_end)
                    location = str(lat_end)+','+str(long_end)
                    print('predicted location is   ')
                    print('*****************')
                    print(location)
                    stamp.append(names) 
                    image.save(names+'_location_'+str(location)+'.jpg')
                    predicted_lat.append(lat_end)
                    predicted_long.append(long_end)
                    locations.append(location) 
                    
                             
                   


api_key = ''

filename = 'cleaned_google_waypoints_demo_1.csv'

points = []
predicted_lat = []
predicted_long = []
way_point_lat = []
way_point_long = []
stamp = []
locations = [] 

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
df.to_csv('cleaned_way_point_test_florid.csv')
                    





for take_pic, direction in zip(drive_predicted_location,bearings):
    meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
    pic_base ='https://maps.googleapis.com/maps/api/streetview?'
    api_key = ''
    fov = '120'
    pitch = '10'
    heading = '1'
    met_params = {'key': api_key,
                  'location':take_pic,

                  }

    pic_params = {'key':api_key,
                  'location':take_pic,
                  'size':"640x640",
                  'heading':heading,
                  'fov':fov,
                  'pitch':pitch,
                  #its likely i will decrease the radius here because if i make the 
                  #adjustments i am making the latlongs segments should be more accurate
                  #and closer to the road.  
                  'radius':'20'
                  }

    meta_response = requests.get(meta_base, params=met_params)
    x = meta_response.json()
    try:
        lat_start = x['location']['lat']
        lng_start = x['location']['lng']
        meta_response = requests.get(meta_base, params=met_params)
    except Exception as e:
        pass

    #print(meta_response.json())
    time.sleep(1)
    
    pictures = str(lat_start)+(str(lng_start)+'jpg')
    
    pic_response = requests.get(pic_base, params=pic_params)

    with open('test.jpg', 'wb') as file:
        file.write(pic_response.content)
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        names = str(timestamp)
        # remember to close the response connection to the API
        pic_response.close()
        time.sleep(2)
        bests = 78
        score_check = [99.999999976]
        thinking = []
        
        image = Image.open('test.jpg')
      
        image_processor = AutoImageProcessor.from_pretrained("/home/j/Desktop/all_images/detr-5k_200e_1e-5_1e-4/"+str(bests))
        model = AutoModelForObjectDetection.from_pretrained("/home/j/Desktop/all_images/detr-5k_200e_1e-5_1e-4/"+str(bests))

        resulting = []
        boxes = []
        scores = []
        both = []
        image_name = []
        
        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            #here are the results they are going into a dictionary 
            results = image_processor.post_process_object_detection(outputs, threshold=0.00000001,
            target_sizes=target_sizes)[0]
            resulting.append(results)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]

            boxes.append(box)
            scores.append(score)
            boths = box , score.item()
            both.append(boths)
            
            
        filtered = sorted(both, key=lambda x:x[1])
        x = filtered[-1:]
        best_box = []
        if len(x) > 0:
            print('found a pole')
 
            print(x)
            print(lat_start,lng_start)
            for bbox in x:
            #print(bbox[1])
                score = 100.0000 - bbox[1]
                best_box.append(bbox[0])
                draw = ImageDraw.Draw(image)

                for box in best_box :
                    x, y, x2, y2 = tuple(box)
                    draw.rectangle((x, y, x2, y2), outline="red", width=1)
                    cropped_image = image.crop((x, y, x2, y2))
                    #this is cutting out and saving the cropped part of the image
                    #cropped_image.save(files)
                    google_camera_focal_length = 5.1
                    #distance = (focal_length * object_height) / image_height 
                    pic_height = 640
                    
                    object_width = cropped_image.size[0]
                    object_height = cropped_image.size[1]
                    #print(object_height)
                    
                    distance = (google_camera_focal_length * pic_height) / object_height
                    x_img= 640
                    y_img = 0
                    area = cropped_image.size[0] * cropped_image.size[1]
                    #print(area)

                    angle_from_camera = ((x - 320) *.28125) + int(heading)
                    print('found a pole.  it is this distance ' + '     '+str(distance)+'  from   ')
                    print(lat_start,lng_start) 
                    print('and at an angle of  '+ str(angle_from_camera))
                    print('  from the camera')
                    
                    R=6371
                    
                    distance = distance * .0002
                    
                    lat1 = radians(lat_start)
                    lon1 = radians(lng_start)
                    a = radians(angle_from_camera)
                    lat2 = asin(sin(lat1) * cos(distance/R) + cos(lat1) * sin(distance/R) * cos(a))
                    lon2 = lon1 + atan2(
                        sin(a) * sin(distance/R) * cos(lat1),
                        cos(distance/R) - sin(lat1) * sin(lat2)
                    )
                    lat_end = degrees(lat2)
                    long_end = degrees(lon2)
                    #print(lat_end, long_end)
                    location = str(lat_end)+','+str(long_end)
                    print('predicted location is   ')
                    print('*****************')
                    print(location)
                    stamp.append(names) 
                    image.save(names+'_location_'+str(location)+'.jpg')
                    predicted_lat.append(lat_end)
                    predicted_long.append(long_end)
                    locations.append(location) 
                    
                    


          
## hi I am the start of the end of this project
## I will be the file that is cleaned and ready to use
## Good job getting this far

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
df.to_csv('cleaned_way_point_test_florid.csv')
                    





for take_pic, direction in zip(drive_predicted_location,bearings):
    meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
    pic_base ='https://maps.googleapis.com/maps/api/streetview?'
    api_key = ''
    fov = '120'
    pitch = '10'
    heading = '180'
    met_params = {'key': api_key,
                  'location':take_pic,

                  }

    pic_params = {'key':api_key,
                  'location':take_pic,
                  'size':"640x640",
                  'heading':heading,
                  'fov':fov,
                  'pitch':pitch,
                  #its likely i will decrease the radius here because if i make the 
                  #adjustments i am making the latlongs segments should be more accurate
                  #and closer to the road.  
                  'radius':'20'
                  }

    meta_response = requests.get(meta_base, params=met_params)
    x = meta_response.json()
    try:
        lat_start = x['location']['lat']
        lng_start = x['location']['lng']
        meta_response = requests.get(meta_base, params=met_params)
    except Exception as e:
        pass

    #print(meta_response.json())
    time.sleep(1)
    
    pictures = str(lat_start)+(str(lng_start)+'jpg')
    
    pic_response = requests.get(pic_base, params=pic_params)

    with open('test.jpg', 'wb') as file:
        file.write(pic_response.content)
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        names = str(timestamp)
        # remember to close the response connection to the API
        pic_response.close()
        time.sleep(2)
        bests = 78
        score_check = [99.999999976]
        thinking = []
        
        image = Image.open('test.jpg')
      
        image_processor = AutoImageProcessor.from_pretrained("/home/j/Desktop/all_images/detr-5k_200e_1e-5_1e-4/"+str(bests))
        model = AutoModelForObjectDetection.from_pretrained("/home/j/Desktop/all_images/detr-5k_200e_1e-5_1e-4/"+str(bests))

        resulting = []
        boxes = []
        scores = []
        both = []
        image_name = []
        
        with torch.no_grad():
            inputs = image_processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            #here are the results they are going into a dictionary 
            results = image_processor.post_process_object_detection(outputs, threshold=0.00000001,
            target_sizes=target_sizes)[0]
            resulting.append(results)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]

            boxes.append(box)
            scores.append(score)
            boths = box , score.item()
            both.append(boths)
            
            
        filtered = sorted(both, key=lambda x:x[1])
        x = filtered[-1:]
        best_box = []
        if len(x) > 0:
            print('found a pole')
 
            print(x)
            print(lat_start,lng_start)
            for bbox in x:
            #print(bbox[1])
                score = 100.0000 - bbox[1]
                best_box.append(bbox[0])
                draw = ImageDraw.Draw(image)

                for box in best_box :
                    x, y, x2, y2 = tuple(box)
                    draw.rectangle((x, y, x2, y2), outline="red", width=1)
                    cropped_image = image.crop((x, y, x2, y2))
                    #this is cutting out and saving the cropped part of the image
                    #cropped_image.save(files)
                    google_camera_focal_length = 5.1
                    #distance = (focal_length * object_height) / image_height 
                    pic_height = 640
                    
                    object_width = cropped_image.size[0]
                    object_height = cropped_image.size[1]
                    #print(object_height)
                    
                    distance = (google_camera_focal_length * pic_height) / object_height
                    x_img= 640
                    y_img = 0
                    area = cropped_image.size[0] * cropped_image.size[1]
                    #print(area)

                    angle_from_camera = ((x - 320) *.28125) + int(heading)
                    print('found a pole.  it is this distance ' + '     '+str(distance)+'  from   ')
                    print(lat_start,lng_start) 
                    print('and at an angle of  '+ str(angle_from_camera))
                    print('  from the camera')
                    
                    R=6371
                    
                    distance = distance * .0002
                    
                    lat1 = radians(lat_start)
                    lon1 = radians(lng_start)
                    a = radians(angle_from_camera)
                    lat2 = asin(sin(lat1) * cos(distance/R) + cos(lat1) * sin(distance/R) * cos(a))
                    lon2 = lon1 + atan2(
                        sin(a) * sin(distance/R) * cos(lat1),
                        cos(distance/R) - sin(lat1) * sin(lat2)
                    )
                    lat_end = degrees(lat2)
                    long_end = degrees(lon2)
                    #print(lat_end, long_end)
                    location = str(lat_end)+','+str(long_end)
                    print('predicted location is   ')
                    print('*****************')
                    print(location)
                    stamp.append(names) 
                    image.save(names+'_location_'+str(location)+'.jpg')
                    predicted_lat.append(lat_end)
                    predicted_long.append(long_end)
                    locations.append(location) 
                    
                    


df = pd.DataFrame({'time_stamp':stamp, 'predicted_lat':predicted_lat, 'predicted_long':predicted_long,'location':locations})
#df1 = df.drop_duplicates()
df.to_csv('09_demo_2_predicted_lat_and_long.csv')
                    
                    
                  

            
            









            
            









                  

            
            









            
            









