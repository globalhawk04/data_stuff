## to scrape the legislatures of the united states and texas into a json file
## i will be applying this to usda
## all of this will go into a database accesible by api that will be tune to llms
### I have scuccessfuly scraped a main list into a csv and will now go through each page to create a json file
#### things I want to hav:
# sponsor
# Last Action 
# committees
# public law No 
# Public Law text
# I am debating on where grabbing the introduced bills to compare them to the final bill
# WHile this would be a fun exncersise I think i need to just stick to what i am doing 
# as I have already done that in previous things




import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import json
import sys


## open csv and create list



List_of_passed_cgov_laws = 'list_passed_bills.csv'
converted_cgov = []


#opening list and getting the website addresses
with open (List_of_passed_cgov_laws,'r') as cfile:
    #skips the first row 
    next(cfile)
    file_reader = csv.reader(cfile)    
    for c_gov_list in file_reader:
        #this is a faster way than a double loop
        get_list = c_gov_list[1]
        #making a list of websites 
        x = get_list.split('?')
        #print(x)
        add_list = 'https://www.congress.gov'+x[0]+'/text/enr'
        converted_cgov.append(add_list)

print(converted_cgov[100])


from selenium import webdriver
import time
from bs4 import BeautifulSoup 
from selenium.webdriver.chrome.options import Options


#option = webdriver.ChromeOptions()
#driver = webdriver.Chrome('/chromedriver',options=option)

driver = webdriver.Chrome()

options = webdriver.ChromeOptions()
#you cannot go headless on this or you will not pass as human and get locked out
#options.add_argument("--headless=new")
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')

driver = webdriver.Chrome(options=options)

#https://www.congress.gov/bill/103rd-congress/house-bill/898/text/enr


all_passed_bills = []

enrolled_bill_text = []
bill_info = []

try:

    for bills in converted_cgov:
        print(bills)

        driver.get(str(bills))
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        #for link in soup.find_all('div', class_='search-row'):
            #print(link)
        # this one only finds hrefs which is goof=d but not useful at the moment

        #for div_class_overview in soup.find_all('div', class_='overview'):
         #   for href in div_class_overview.find_all('a'):
          #      #print(href['href'])
           #     introduced = href.text
            #    #print(introduced)
             #   #all_passed_bills.append(href['href'])

        # this one gobbles up everything in the overview table i can parse out what I want later
        for td in soup.find_all('table', class_='standard01'):
            #print(td.text)
            x = td.text
            if len(x) > 0:
                cleaned_text = ' '.join(x.split())
                bill_info.append(cleaned_text)
            else:
                sys.exit()

          #  all_passed_bills.append(td.text)

            #or href in div_class_overview.find_all('a'):
                #print(href['href'])
                #introduced = href.text
                #print(introduced)
                #all_passed_bills.append(href['href'])
        #test_list = list(set(all_passed_bills))

        


        for erb_text in soup.find_all('div', class_='generated-html-container enrolled-bill'):
            #print(erb_text.text)
            x = erb_text.text
            cleaned_text = ' '.join(x.split())
            #print(cleaned_text)
            enrolled_bill_text.append(cleaned_text)

        create_json = []

        for tags, text in zip(bill_info,enrolled_bill_text):

            create_json.append(dict(
            {
                'tags':tags,
                'text': text
            }))
        json_data = json.dumps(create_json, indent=4)
        with open('cgov_data_tracker_final.json','w') as f:
            f.write(json_data)
    

        time.sleep(8)

    # 


except Exception as e:
    print(e)
    pass                








#convert to json
create_json = []

for tags, text in zip(bill_info,enrolled_bill_text):

    create_json.append(dict(
    {
        'tags':tags,
        'text': text
    }))
json_data = json.dumps(create_json, indent=4)
with open('cgov_data_final.json','w') as f:
    f.write(json_data)
    

#print(all_passed_bills)
#print(len(all_passed_bills))
#print(len(test_list))

#df = pd.DataFrame({'list_passed_bills':test_list})
#df.to_csv('list_passed_bills.csv') 



## in the og program I ran range loop for inside each legis session.  
## I can do a better job by using a series of loops and lists to go through everything alls at once







### loop set goes here'
'''
URL_house = 'https://www.congress.gov/bill/118th-congress/house-bill/3944/text/pl'


public_law = 'pl'
reported_to_senate = 'rs'
referred_in_senate = 'rfs'
engrossed_in_house = 'eh'
introduced_in_house = 'ih'


URL_senate ' https://www.congress.gov/bill/110th-congress/senate-bill/3001/text/pl'



public_law = 'pl'
engrossed_in_senate = 'es'
engrossed_amendment_house = 'eah'
placed_on_calendar_senate = 'pcs'

page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")



#need to use an xml parser too here is something to start
# soup = BeautifulSoup(page.content, "lxml")

reults = soup.find(id='billTextContainer')
#div class = 'generated-html-container'

try:
    with open ('file_name.txt', 'w', encoding='utf-8') as f_out:
        f_out.write(results.prettify())
        #writed this out in just a regular way
        # f_out.write(results)

except Exception as e():
    print(e)

time.sleep(1)



# the most important thing here is getting it all into a json file. which I should have all that stuff in tamu_h

#get a list of all bills that have come into law from 1991 to present

'https://www.congress.gov/quick-search/legislation?wordsPhrases=&wordVariants=on&congressGroups%5B0%5D=0&congresses%5B0%5D=all&legislationNumbers=&legislativeAction=&sponsor=on&representative=&senator=&q=%7B%22congress%22%3A%5B118%2C117%2C116%2C115%2C114%2C113%2C112%2C111%2C110%2C109%2C108%2C107%2C106%2C105%2C104%2C103%2C%22102%22%5D%2C%22bill-status%22%3A%22law%22%7D'

'https://www.congress.gov/quick-search/legislation?wordsPhrases=&wordVariants=on&congressGroups%5B0%5D=0&congresses%5B0%5D=all&legislationNumbers=&legislativeAction=&sponsor=on&representative=&senator=&q=%7B%22congress%22%3A%5B118%2C117%2C116%2C115%2C114%2C113%2C112%2C111%2C110%2C109%2C108%2C107%2C106%2C105%2C104%2C103%2C%22102%22%5D%2C%22bill-status%22%3A%22law%22%7D'

'https://www.congress.gov/quick-search/legislation?wordsPhrases=&wordVariants=on&congressGroups[0]=0&congresses[0]=all&legislationNumbers=&legislativeAction=&sponsor=on&representative=&senator=&q=%7B%22congress%22%3A%5B118%2C117%2C116%2C115%2C114%2C113%2C112%2C111%2C110%2C109%2C108%2C107%2C106%2C105%2C104%2C103%2C%22102%22%5D%2C%22bill-status%22%3A%22law%22%7D&pageSize=250'
'https://www.congress.gov/quick-search/legislation?wordsPhrases=&wordVariants=on&congressGroups[0]=0&congresses[0]=all&legislationNumbers=&legislativeAction=&sponsor=on&representative=&senator=&q=%7B%22congress%22%3A%5B118%2C117%2C116%2C115%2C114%2C113%2C112%2C111%2C110%2C109%2C108%2C107%2C106%2C105%2C104%2C103%2C%22102%22%5D%2C%22bill-status%22%3A%22law%22%7D&pageSize=250'




from bs4 import BeautifulSoup
import requests
import pandas as pd
response= requests.get('https://ofsistorage.blob.core.windows.net/publishlive/ConList.html')
soup = BeautifulSoup(response.text, 'lxml')
name_list = soup.find('body')
name_list_items = name_list.find_all('ol')
data = []

for name in name_list_items:
    list_items = name.find_all('li') 
    list_items = [item.text for item in list_items]
    data.append(list_items)

df = pd.DataFrame(data)
print(df)
'''

## to get the span class

'''
# parse the html
soup = BeautifulSoup(html_doc)

# find a list of all span elements
spans = soup.find_all('span', {'class' : 'blue'})

# create a list of lines corresponding to element texts
lines = [span.get_text() for span in spans]

# collect the dates from the list of lines using regex matching groups
found_dates = []
for line in lines:
    m = re.search(r'(\d{2}/\d{2}/\d{2} \d+:\d+[a|p]m)', line)
    if m:
        found_dates.append(m.group(1))

# print the dates we collected
for date in found_dates:
    print(date)

'''


#ok so what i am going to do here is scrape the list of all passed laws from 1991 to present
# to do this i need to navigate the ol class to get the number of bills displayed on page then get the href for the bill and go from there.
# i don't know if I can do it this way so I am going to try and see



#name_list = soup.find('body')
#name_list_items = soup.find_all('ol')
#print(name_list_items)


#data = []

#for name in name_list_items:
#    list_items = name.find_all('li') 
#    list_items = [item.text for item in list_items]
#    data.append(list_items)

#df = pd.DataFrame(data)
#print(df)
'''
import requests
from bs4 import BeautifulSoup as soup
import webbrowser
import os
from urllib.request import Request, urlopen
import pandas as pd
import time
import csv
import json 


websiteurl = 'https://www.congress.gov/quick-search/legislation?wordsPhrases=&wordVariants=on&congressGroups%5B0%5D=0&congresses%5B0%5D=all&legislationNumbers=&legislativeAction=&sponsor=on&representative=&senator=&q=%7B%22congress%22%3A%5B118%2C117%2C116%2C115%2C114%2C113%2C112%2C111%2C110%2C109%2C108%2C107%2C106%2C105%2C104%2C103%2C%22102%22%5D%2C%22bill-status%22%3A%22law%22%7D&pageSize=250'
#print(websiteurl)
try:
    req = Request(websiteurl, headers={"User-Agent": 'Mozilla/115.0'})
    webpage = urlopen(req).read()
    soups = soup(webpage,'html.parser')
    print(soups)
    #for link in soups.find_all('div', class_ = 'search-column-main basic-search-results nav-on'):
        #print(link)
        #for price in link.find_all('span', class_ = 'product-price-value'):
            #print(price.text)
    #for link in soups.find_all('h1'):
        #print(link.text)

        

except Exception as e:
    print(e)






from selenium import webdriver
import time
from bs4 import BeautifulSoup 
from selenium.webdriver.chrome.options import Options


#option = webdriver.ChromeOptions()
#driver = webdriver.Chrome('/chromedriver',options=option)

driver = webdriver.Chrome()

options = webdriver.ChromeOptions()
#you cannot go headless on this or you will not pass as human and get locked out
#options.add_argument("--headless=new")
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')

driver = webdriver.Chrome(options=options)



all_passed_bills = []

try:
    for page in range(1,29):
        print(page)
        driver.get('https://www.congress.gov/quick-search/legislation?wordsPhrases=&wordVariants=on&congressGroups%5B0%5D=0&congresses%5B0%5D=all&legislationNumbers=&legislativeAction=&sponsor=on&representative=&senator=&q=%7B%22congress%22%3A%5B118%2C117%2C116%2C115%2C114%2C113%2C112%2C111%2C110%2C109%2C108%2C107%2C106%2C105%2C104%2C103%2C%22102%22%5D%2C%22bill-status%22%3A%22law%22%7D&pageSize=250&page='+str(page))
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        #for link in soup.find_all('div', class_='search-row'):
            #print(link)
        for span_class in soup.find_all('span', class_='result-heading'):
            for href in span_class.find_all('a'):
                #print(href['href'])
                all_passed_bills.append(href['href'])

    test_list = list(set(all_passed_bills))
    time.sleep(1)


except Exception as e:
    print(e)
    pass                


print(len(all_passed_bills))
print(len(test_list))

df = pd.DataFrame({'list_passed_bills':test_list})
df.to_csv('list_passed_bills.csv') 




#need to loop this in a range from 0 to 28 to get all the bills 


driver.get('https://www.congress.gov/quick-search/legislation?wordsPhrases=&wordVariants=on&congressGroups%5B0%5D=0&congresses%5B0%5D=all&legislationNumbers=&legislativeAction=&sponsor=on&representative=&senator=&q=%7B%22congress%22%3A%5B118%2C117%2C116%2C115%2C114%2C113%2C112%2C111%2C110%2C109%2C108%2C107%2C106%2C105%2C104%2C103%2C%22102%22%5D%2C%22bill-status%22%3A%22law%22%7D&pageSize=250')

driver.get('https://www.congress.gov/quick-search/legislation?wordsPhrases=&wordVariants=on&congressGroups%5B0%5D=0&congresses%5B0%5D=all&legislationNumbers=&legislativeAction=&sponsor=on&representative=&senator=&q=%7B%22congress%22%3A%5B118%2C117%2C116%2C115%2C114%2C113%2C112%2C111%2C110%2C109%2C108%2C107%2C106%2C105%2C104%2C103%2C%22102%22%5D%2C%22bill-status%22%3A%22law%22%7D&pageSize=250&page=1')

title = driver.title

print(title)



page_source = driver.page_source
soup = BeautifulSoup(page_source, 'html.parser')

#print(soup.text)

for link in soup.find_all('div', class_='search-row'):
    #print(link)
    for span_class in link.find_all('span', class_='result-heading'):
        for href in span_class.find_all('a'):
            #print(href)
            print(href.text)
    for span_class in link.find_all('span', class_='result-title bottom-padding'):
        print(span_class.text)




    #this did not work but I found the ahref for each page so that is what will work.  

    ##print(a.get('href'))

'''
