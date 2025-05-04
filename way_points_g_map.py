## testing gmap API for routes

import json
import requests
import urllib
import re
api_key = ''



import googlemaps

# Create a Google Maps API object
gmaps = googlemaps.Client(key=api_key)

# Set the origin and destination
origin = ""
destination = ''
waypoint0 = ''
waypoint1 = ''
waypoint2 = ''
waypoint3 = ''
waypoint4 = ''
waypoint5 = ''

waypoint = [waypoint0,waypoint1,waypoint2,waypoint3,waypoint4,waypoint5]
# Compute the route
route = gmaps.directions(origin, destination,waypoints=waypoint)

# Print the route
for key in route:
    #print(key['legs'])
    for keys in key['legs']:
        #print(keys['duration'])
        #print(keys['steps'])

        for keying in keys['steps']:
            
            #print(keying.get('html_instructions'))
            if "Turn" or "Head" in keying.get('html_instructions'):
                text = keying.get('html_instructions')
                # Find all instances of <b>bold</b> text
                match = re.search(r"(<b>.*?</b>)", text)
                if 'left' or 'right' or 'straight' or 'northwest' or 'north' or'northeast' or 'south' or 'southwest' or 'southeast' or 'east' or 'west' in match.group(1):
                    print(keying.get('distance')['value'])
                    print(keying.get('start_location'))
        
json_data = json.dumps(route, indent=4)

with open('gmap_geo.json','w') as f:
	f.write(json_data)

