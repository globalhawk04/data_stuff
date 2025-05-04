##


#https://capitol.texas.gov/tlodocs/89R/billtext/pdf/HB00020I.pdf#navpanes=0
#https://capitol.texas.gov/tlodocs/89R/billtext/pdf/HB00001I.pdf#navpanes=0

import urllib
import urllib.request
from bs4 import BeautifulSoup
import os
import re
import string
import json
import time



house_list_of_urls = []
create_json = []
all_keys = []


# this code will let me open previous json scan filed bills and 
# compare to what is alread in json then 
# crawl the ones it needs to add
# i have not added a append json to this but i need to run it tonigh 
# and its a good start
#f = open('All_Texas_Senate_Bills.json')
#data = json.load(f)
#for key in data:
	#print(key.keys())
	#print(key['Bill Main Url'])
	#all_keys.append(key['Bill Main Url'])

#difference = list(set(house_list_of_urls) - set(all_keys))
#print(len(difference))

website ='https://capitol.texas.gov/Reports/Report.aspx?LegSess=89R&ID=senatefiled'
def make_soup1(url):
	thepage = urllib.request.urlopen(url)
	soupdata = BeautifulSoup(thepage, "lxml")
	return soupdata


soup = make_soup1(website)
time.sleep(1)
for record in soup.findAll('tr'): #type of data
	for data in record.findAll('a'):# type of data from website
		#print(data.get('href'))
		house_list_of_urls.append(data.get('href'))

#print(len(house_list_of_urls))

for list_of_urls in house_list_of_urls:
	print(list_of_urls)
	time.sleep(1)

	try:
		hrbillsaved=[] # saves everything into a list
		soup = make_soup1(list_of_urls)# the website i am scraping from
		time.sleep(1)
		for bill_num in soup.findAll('span',id='usrBillInfoTabs_lblBill'):
			print(bill_num.text)
			bill_name = bill_num.text

		for data in soup.findAll('td', id = 'cellAuthors'):# type of data from website
			print(data.text)
			author = data.text

		for data in soup.findAll('td', id = 'cellCaptionText'):# type of data from website
			print(data.text)
			caption = data.text
		committees =[]
		for data in soup.findAll('td', id = 'cellComm1Committee'):# type of data from website
			for href in data.findAll('a'):
				if href is None:
					print('no committees yet')
					pass
				else:
					print('https://capitol.texas.gov/Committees/'+href.get('href'))
					committees.append('https://capitol.texas.gov/Committees/'+href.get('href'))

		for data in soup.findAll('td', id = 'cellLastAction'):# type of data from website
			print(data.text)
			last_action = data.text

		for record in soup.findAll('tr'): #type of data
			for data in record.findAll('a', class_ = 'enabledButNotActive'):# type of data from website
				if 'Text' in data.get('href'):
					bill_text_url ='https://capitol.texas.gov/BillLookup/'+data.get('href')
					time.sleep(1)
					soup = make_soup1(bill_text_url)
					for record in soup.findAll('tr'): #type of data
						for data in record.findAll('a'):
							##print(data.get('href'))
							if data.get('href') is None:
								##print('none')
								pass
							elif 'html' in data.get('href'):
								#time.sleep(1)
								#print(data.get('href'))
								bill_text_html = 'https://capitol.texas.gov/'+data.get('href')
								time.sleep(1)
								soup = make_soup1(bill_text_html)# the website i am scraping from
								for record in soup.findAll('tr'): #type of data
									for data in record.findAll('td'): # type of data from website
										##print(data.text)
										data.text.strip()
										hrbillsaved.append(data.text)
								bill = ' '.join(hrbillsaved)
								bill_strip = bill.strip()
								##print(bill_strip)

								create_json.append(dict(
									{'Bill Name': bill_name,
									 'Bill Main Url': list_of_urls,
									 'Author': author,
									 'Caption': caption,
									 'Commitee': committees,
									 'Last Action': last_action,
									 'Bill Text Url': bill_text_url,
									 'Bill Text HTML': bill_text_html,
									 'Bill Text': bill
									}))
			#for data in record.findAll('a'):
				##print(data.get('href')
	except Exception as e:
		print(str(e))
		pass
	json_data = json.dumps(create_json, indent=4)

	with open('New_All_Texas_Senate_Bills.json','w') as f:
		f.write(json_data)

