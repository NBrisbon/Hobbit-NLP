#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.text import Text
from nltk.corpus import brown
import requests
from bs4 import BeautifulSoup
nltk.download('popular')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[3]:


r = requests.get('https://archive.org/stream/TheHobbitByJRRTolkienEBOOK/The%20Hobbit%20byJ%20%20RR%20Tolkien%20EBOOK_djvu.txt')

# Setting the correct text encoding of the HTML page
r.encoding = 'utf-8'

# Extracting the HTML from the request object
html = r.text

# Printing the first 2000 characters in html
print(html[:2000])


# In[173]:


# Creating a BeautifulSoup object from the HTML
soup = BeautifulSoup(html, 'html.parser')

# Getting the text out of the soup
text = soup.get_text()

# Entire Book
book = text[9515:531263]

# An Unexpected Party
ch1 = book[0:47620] 

# Roast Mutton
ch2 = book[47625:76488]

# A Short Rest
ch3 = book[76493:92164]

# Over Hill and Under Hill
ch4 = book[92169:114853]

# Riddles in the Dark
ch5 = book[114858:152917]

# Out of the Frying-Pan into the Fire
ch6 = book[152924:189738]

# Queer Lodgings
ch7 = book[189744:238305]

# Flies And Spiders
ch8 = book[238311:293647]

# Barrels Out Of Bond
ch9 = book[293653:325083]

# A Warm Welcome
ch10 = book[325089:346565]

# On The Doorstep
ch11 = book[346570:362945]

# Inside Information 
ch12 = book[362950:401813]

# Not At Home
ch13 = book[401819:441166]

# Fire And Water
ch15 = book[441172:423379]

print(ch15)


# In[174]:


book.find("Chapter XIV")


# In[ ]:




