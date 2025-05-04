#searching the json to find tags

import json
import pandas as pd

# Opening JSON file
f = open('data.json')

# returns JSON object as
# a dictionary
data = json.load(f)


how_many = input('how many key words you looking for? ')

search_terms = []

for x in range(0,int(how_many)):
    user_input = input('enter a search term: ')
    search_terms.append(user_input)
    
print(search_terms)

names = []
num_matches = []
match_words = []
email = []

names_all = []
num_matches_all = []
match_words_all = []
email_all = []


for key in data:
    store = 0
    matches = []
    for key_words in search_terms:
        if key_words.lower() in key['text']:
            store +=1
            matches.append(key_words)

    if store == int(how_many):
        pass
        names_all.append(key['name'])
        num_matches_all.append(store)
        match_words_all.append(' '.join(matches))
    if store > 0:
        names.append(key['name'])
        email.append(key['email'])
        num_matches.append(store)
        match_words.append(' '.join(matches))
        
print(names,num_matches,match_words)    

df = pd.DataFrame({'User':names,'email':email,'matches':num_matches,'words':match_words})
df.to_csv(str(search_terms)+'_search_results.csv') 
               




