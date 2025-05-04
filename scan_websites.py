import pandas as pd
import csv
import requests
import string
import json
import requests
from bs4 import BeautifulSoup 
from urllib.request import Request, urlopen
import itertools 

tamu_h_stud_list = 'HonorStud - New List.csv'
email = []
e_port = []
first_name = []
last_name = []
stud_site_error = [] 
all_stud_list = []
all_stud_list_ref = []
all_stud_crawl_url = []
site_error = []
site_edit = []

with open (tamu_h_stud_list,'r') as cfile:
    next(cfile)
    file_reader = csv.reader(cfile)    
    for _list in file_reader:
        first_name.append(_list[0]) 
        last_name.append(_list[1])
        e_port.append(_list[2])
        email.append(_list[3])
print('********************')

for e_port, email, first_name, last_name in zip(e_port,email,first_name,last_name):
    if '//' not in e_port:
        e_port = 'https://'+e_port  
    if 'edit' in e_port:
        e = e_port
        f = first_name
        l = last_name
        em = email
        site_edit.append([f,l,em,e])
        pass
    try:
        url = e_port
        print(url)
        if 'http://honors.tamu.edu' in url:
            print('Honors Website')
            pass
        req = Request(url, headers={"User-Agent": 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        soups = BeautifulSoup(webpage,'html.parser')
        for link in soups.find_all('a'):
            x = link.get('href')    
            if '.html' in x:
                all_stud_list.append([url+x,first_name,last_name,email]) 
            if 'https://support.google.com' in x:
                print('support')  
            elif '/edit' in x:
                print('site is not open')
            elif 'http://honors.tamu.edu' in x:
                print('Honors Website')                
            elif 'https://sites.google.com' in x:
                print('sites')
            elif 'https://accounts.google' in x:
                print('accounts')
            elif 'https://www.facebook.com' in x:
                print('facebook')
            elif 'https://www.twitter.com' in x:
                print('twitter')
            elif 'https://www.instagram.com' in x:
                print('instagram') 
            elif 'https://www.wix.com' in x:   
                print('wix link')                 
            elif 'mailto' in x:
                print('email address')
            elif 'http://www.facebook.com' in x:
                print('facebook')
            elif 'http://www.twitter.com' in x:
                print('twitter')
            elif 'https://www.instagram.com' in x:
                print('instagram') 
            elif 'https://www.linkedin.com' in x:
                print('linkedin')  
            elif 'http://www.linkedin.com' in x:
                print('linkedin')  
            elif 'http://www.wix.com' in x:
                print('wix links')    
            elif 'https://www.canvastatus.com' in x:
                print('canvastatus')                      
            elif 'https://www.godaddy.com/' in x:
                print('go daddy')                       
            elif 'http://wix.com/?utm_campaign=vir_created_with' in x:
                print('wix utm')   
            elif 'http://wix.com?utm_campaign=vir_created_with' in x:
                print('wix utm')          
            elif 'http://youtube.com/user/Wix' in x:
                print('wix youtube')                       
            elif 'https://tamureal.com/' in x:
                print('tamu real')                      
            elif 'https://twitter.com/wix' in x:
                print('wix twitter')                       
            elif 'https://twitter.com/Wix' in x:
                print('wix twitter')                       
            elif 'https://www.aggiedsp.org' in x:
                print('aggiedsp')   
            elif 'https://instagram.com/wix/' in x:
                print('instagram')   
            elif 'https://mays.tamu.edu/office-of-diversity-and-inclusion/rap/ ' in x:
                print('mays')   
            elif 'https://alot.tamu.edu/' in x:
                print('alot')                                                            
            elif 'https://neocities.org/' in x:
                print('neo cities')                                               
            elif 'https' not in x:
                if '//www.wix.com'  in x:
                    print('wix links')
                elif '/view/' in x:
                    urls = x.split('/view/')
                    x = '/view/'+urls[1]
                    url_main = url.split('/view/')
                    y = url_main[0]
                    all_stud_list.append([y+x,first_name,last_name,email])
                elif '/tamu.edu/' in x:
                    urls = x.split('/tamu.edu/')
                    x = '/tamu.edu/'+urls[1]
                    url_main = url.split('/tamu.edu/')
                    y = url_main[0]
                    all_stud_list.append([y+x,first_name,last_name,email])
                elif '/' in x:
                    if '.com/' in url:
                        url_main = url.split('com/')
                        y = url_main[0]+ 'com'
                        #print(y+x)
                        all_stud_list.append([y+x,first_name,last_name,email])
                else:
                    all_stud_list.append([url+x,first_name,last_name,email]) 
            else:
                all_stud_list.append([x,first_name,last_name,email])  
    except Exception as e:
        stud_site_error.append(url)

all_stud_list.sort()
sorted_all_studs = list(all_stud_list for all_stud_list,_ in itertools.groupby(all_stud_list))

sorted_first = []
sorted_last = []
sorted_link = []
sorted_email = []
for sorted_links in sorted_all_studs:
    sorted_link.append(sorted_links[0])
    sorted_last.append(sorted_links[1])
    sorted_first.append(sorted_links[2])
    sorted_email.append(sorted_links[3])

df = pd.DataFrame({'crawl_url':sorted_link,'last_name':sorted_last,'first_name':sorted_first,'email':sorted_email})

main_list_with_email = []
main_list_with_text = []
main_list_with_url = []
main_list_first_name = []
main_list_last_name = []

print('Grabbing Text from Websites')

for row in df.itertuples():
    get_list = str(row.crawl_url)
    get_last_name = str(row.last_name)
    get_first_name = str(row.first_name)
    get_email = str(row.email)
    try:
        email = get_email
        url = get_list
        last_name = get_last_name
        first_name =  get_first_name
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, 'html.parser')
        for words in soup.find_all('div', class_ = 'paragraph'):
            x = words.text
            print(x)
            main_list_with_text.append(x)
            main_list_with_url.append(url)
            main_list_with_email.append(email)
            main_list_last_name.append(last_name)
            main_list_first_name.append(first_name)
        for words in soup.find_all('p'):
            x = words.text
            print(x)
            main_list_with_text.append(x)
            main_list_with_url.append(url)
            main_list_with_email.append(email)
            main_list_last_name.append(last_name)
            main_list_first_name.append(first_name)
    except Exception as e:
        stud_site_error.append(url)

df=pd.DataFrame({'url':main_list_with_url,'email':main_list_with_email,'text':main_list_with_text,'first':main_list_first_name,'last':main_list_last_name})

print('removing punctuation and stop words')

tag = []
string_words_removed = []
fn = []
ln = []
stop_words = ["|",',','.','?','!','',"i", "i'm","me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "'s", "'t", "can", "will", "just", "don't", "should", "now","let","click"]

for row in df.itertuples():
    url_x = row[1]
    tag_y = row[2]
    tex_z = row[3]
    fn1 = row[4]
    ln1 = row[5]
    words = tex_z.lower()
    table = str.maketrans('', '',   string.punctuation)
    removed = words.translate(table)
    stop_remove = [word for word in words.split() if word not in stop_words]
    new_text = " ".join(stop_remove)
    string_words_removed.append(new_text)
    tag.append(tag_y)
    fn.append(fn1)
    ln.append(ln1)    

df = pd.DataFrame({'tag':tag,'text':string_words_removed,'fn':fn,'ln':ln})

#convert to json
print('creating json database')

list_all = []
list_of_words_in_file = []
list_of_tags_in_file = []

for words_in_file in df.itertuples():
    list_all.append(words_in_file)
    list_of_words_in_file.append(words_in_file[2])
    list_of_tags_in_file.append(words_in_file[1])
list2 = []

for i in list_of_tags_in_file:
    if i not in list2:
        list2.append(i)
create_json = []
total_list = []

for x in list2:
    tagging = x
    texting = []
    last = ''
    first = ''
    for q,w,z,a,b in list_all:
        if w == x:
            texting.append(z)
            first = a 
            last = b
    list2 = []
    for i in texting:
        if i not in list2:
            list2.append(i)
    give_it = ' '.join(list2)
    create_json.append(dict(
    {
	    'email':tagging,
	    'text': give_it,
        'name': last +' ' + first
    }))
json_data = json.dumps(create_json, indent=4)
with open('tamu_h_student_data.json','w') as f:
	f.write(json_data)
print('All Done')

#create a list of sites in edit mode
site_error_first = []
site_error_last = []
site_error_email = []
site_error_e_port = []
for x in site_edit:
    site_error_first.append(x[0])
    site_error_last.append(x[1])
    site_error_email.append(x[2])
    site_error_e_port.append(x[3])

df = pd.DataFrame({'first_name':site_error_first,'last_name':site_error_last, 'email':site_error_email,'e_port':site_error_e_port})
df.to_csv('list_of_websites_in_edit_mode.csv') 
