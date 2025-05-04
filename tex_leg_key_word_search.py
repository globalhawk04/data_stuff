#searching the json to find tags

import json
import pandas as pd

# Opening JSON file
f = open('New_All_Texas_House_Bills.json')

# returns JSON object as
# a dictionary
data = json.load(f)


how_many = input('how many key words you looking for? ')

search_terms = []

for x in range(0,int(how_many)):
    user_input = input('enter a search term: ')
    search_terms.append(user_input)
    
print(search_terms)

text = []
num_matches = []
match_words = []
url = []
hb_bill_matched = []
hb_bill = []
caption = []
text_all = []
num_matches_all = []
match_words_all = []
url_all = []
author = []
committee = []


for key in data:
    store = 0
    matches = []
    for key_words in search_terms:
        if key_words.lower() in key['Bill Text']:
            store +=1
            matches.append(key_words)

    if store == int(how_many):
        #names_all.append(key['name'])
        #url_all.append(key['hb text url'])
        if key['Bill Main Url'] not in url:
            url.append(key['Bill Main Url'])
            num_matches_all.append(store)
            match_words_all.append(' '.join(matches))
            caption.append(key['Caption'])
            committee.append(key['Commitee'])
            author.append(key['Author'])
    #if store > 0:
        #if key['Bill Main Url'] not in url:
            #url.append(key['Bill Main Url'])
            #num_matches_all.append(store)
            #match_words_all.append(' '.join(matches))
        
print(num_matches_all,url_all)    

print(len(url_all))
#print(hb_bill_matched)
df = pd.DataFrame({'author':author,'committee':committee,'url':url,'matches':num_matches_all,'caption':caption})
df.to_csv(str(search_terms)+'_House_search_results.csv') 
         
'''      
# Opening JSON file
f = open('Texas_Senate_Bills.json')

# returns JSON object as
# a dictionary
data = json.load(f)


how_many = input('how many key words you looking for? ')

search_terms = []

for x in range(0,int(how_many)):
    user_input = input('enter a search term: ')
    search_terms.append(user_input)
    
print(search_terms)

text = []
num_matches = []
match_words = []
url = []
hb_bill_matched = []
hb_bill = []

text_all = []
num_matches_all = []
match_words_all = []
url_all = []


for key in data:
    store = 0
    matches = []
    for key_words in search_terms:
        if key_words.lower() in key['Bill Text']:
            store +=1
            matches.append(key_words)

    if store == int(how_many):
        #names_all.append(key['name'])
        num_matches_all.append(store)
        match_words_all.append(' '.join(matches))
        #url_all.append(key['hb text url'])
        hb_bill_matched.append(key['Bill Main Url'])
    if store > 0:
        hb_bill.append(key['Bill Main Url'])
        #url.append(key['hb text url'])
        #text.append(key['Bill Text'])
        num_matches.append(store)
        match_words.append(' '.join(matches))
        
print(num_matches_all,url_all)    

print(len(url_all))
print(hb_bill_matched)
df = pd.DataFrame({'bill':hb_bill, 'url':url,'matches':num_matches,'words':match_words})


df.to_csv(str(search_terms)+'_senate_search_results.csv') 


'''
