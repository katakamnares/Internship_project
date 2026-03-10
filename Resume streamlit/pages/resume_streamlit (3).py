

import pandas as pd
import numpy as np
import streamlit as st
import docx
import re
import os
import nltk
from PyPDF2 import PdfReader

#File reading utilities
# docx reader

st.markdown("# :red[Resume Classification]")

# Getting the upload file from the session

File = st.session_state.get('file')

def extract_text_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return ""

# pdf reader
def extract_text_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return ""

# generic wrapper
def extract_text_from_file(path):
    if path.name.endswith(".docx"):
        return extract_text_docx(path)
    if path.name.endswith(".pdf"):
        return extract_text_pdf(path)
    # .doc older binary files often not reliably readable - skip or convert externally
    if path.name.endswith(".doc"):
        # best-effort: try docx if file actually docx with wrong extension; else skip
        try:
            return extract_text_docx(path)
        except:
            return ""
    # otherwise return empty
    return ""




text = extract_text_from_file(File)
dic = dict(filename = File.name, text = text)
df = pd.DataFrame(dic,index = [0])

# Data Cleaning and Data Preprocessing Step

def basic_clean(text):
    text = str(text)
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def text_to_lower_onlyalpha(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create dataframe columns: raw, cleaned, tokenized, label
df['text_raw'] = df['text'].apply(basic_clean)
df['text_alpha'] = df['text_raw'].apply(text_to_lower_onlyalpha)
df['word_count'] = df['text_alpha'].apply(lambda x: len(x.split()))
df['extension'] = df['filename'].apply(lambda x: x.split(".")[-1].lower())

st.subheader('Final DataFrame')
st.write(df)

# importing the Pickel and loading the final model

import pickle
file_path = r'C:/Users/HP/Downloads/model3.pkl'
with open(file_path,'rb') as file:
  model_dict = pickle.load(file);

model = model_dict['tfidf']
X = model.transform(df['text_alpha'])

# Prediction the target Value

model = model_dict['model']
y_predict = model.predict(X)

st.markdown("## :green[Predicted Value]")

def predict(y):
    if y == 0:
        return 'PeopleSoft Consultant'
    elif y == 1:
        return 'React Developer'
    elif y == 2:
        return 'SQL Developer'
        
    return 'WorkDay Consultant'

result = predict(y_predict) 
name = os.path.splitext(File.name)[0] 

# Displaying the final prediction value

st.write(f'The {name} is a {result}')