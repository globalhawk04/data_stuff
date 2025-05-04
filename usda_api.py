##Testing Api Stuff
#
import requests
import json
from datetime import datetime
import json

limit = '1000'



api_key = ''

url = 'https://marsapi.ams.usda.gov/services/v1.2/reports'


url = 'https://marsapi.ams.usda.gov/services/v3.0/reports/1602'
from requests.auth import HTTPBasicAuth

resp = requests.get(url, "dataset", auth=HTTPBasicAuth('7XWk8PBs9+P87bCf5SRwHoTFtumvRH96',''))

resps = resp.json()

print(resps)

json_data = json.dumps(resps, indent=4)
with open('cme_weekly_recap.json','w') as f:
	f.write(json_data)
