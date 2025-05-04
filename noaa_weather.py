#





import requests
import json
from datetime import datetime

limit = '1000'



token = ''
#url = "http://www.ncei.noaa.gov/cdo-web/api/v2/stations=WXK31&startdate=2024-01-04&enddate=2024-01-05"


#get all texas stations with and offset 
url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/stations?locationid=FIPS:48&limit=1000&offset=1000'

url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/stations?locationid=FIPS:48&limit=1000'
#url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/stations?locationid=FIPS:48009'

#get daily summaries 
#url = ' https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&stationid=GHCND:USC00418139&startdate=2024-01-01&enddate=2024-01-31&limit=1000'


#NORMAL_DLY
#url = ' https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=NORMAL_DLY&stationid=GHCND:USC00418139'

url = "http://www.ncei.noaa.gov/cdo-web/api/v2/COOP:COOP:419729"


#GHCND



#get all available data sets 
url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/datasets?stationid=	GHCND:USW00013966'

#scotland 
#GHCND:USC00418139

#get specific data set 
url = ' https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&stationid=GHCND:USW00013966&startdate=2024-02-01&enddate=2024-02-05'


headers = {"token":token}

r = requests.get(url, "dataset", headers = headers).text
print(r)
response = json.loads(r)
response_meta = response['metadata']
response = response['results']
#response = response[0]
print(response)
print(response_meta)

with open("daily_wf_GHCND:USW00013966.json", "w") as f:
	json.dump(response, f)
