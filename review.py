#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import streamlit as st
import pickle


# In[4]:


#!pip install textblob_fr 


# In[2]:


#!pip install nltk


# In[1]:


from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re
import string
from textblob import TextBlob


# In[6]:


# Add a title and intro text
st.title('Reviw of Chrome Apps')
st.text('This is a ML model of Chrome Reviews where Semantics of review does not match rating')


# In[12]:


data = pd.read_csv('C:/chrome_reviews.csv')
st.write("To create a table")
df=pd.DataFrame(data)
st.write(df)
st.table(df)


# In[13]:


df.shape


# In[14]:


df.columns


# In[15]:


df.isnull().sum()


# In[17]:


df= df.drop(['User Name'],axis=1)
df= df.drop(['Developer Reply'],axis=1)
df= df.drop(['Version'],axis=1)
df= df.drop(['Review URL'],axis=1)
df= df.drop(['Review Date'],axis=1)
df= df.drop(['App ID'],axis=1)
df.head()
st.write(df)
st.table(df)


# In[23]:


df.dropna(inplace=True)


# In[24]:


df.isnull().sum()


# In[26]:


df.head(10)
st.write(df)
st.table(df)


# In[18]:


df = df[df.Star != 5]
df = df[df.Star != 4]
df = df[df.Star != 3]
df.head()


# In[28]:


senti_list = []
for i in df["Text"]:
    score = TextBlob(i).sentiment[0]
    if (score > 0):
        senti_list.append('Positive')
    elif (score < 0):
        senti_list.append('Negative')
    else:
        senti_list.append('Neutral') 


# In[29]:


df["sentiment"]=senti_list
df.head()
st.write(df)
st.table(df)


# In[37]:


df = df[df.sentiment == 'Positive']
df.head(10)
st.write(df)
st.table(df)


# In[38]:


df.to_csv(r'C:\Users\kripa\OneDrive\Documents\datanew.csv',index=False)
st.write(df)
st.table(df)


# In[ ]:




