import re
import pandas as pd

def clean_text(text):
    text=text.lower()
    text=re.sub(r"http\S+","",text)
    text=re.sub(r"<.*?>","",text)
    text=re.sub(r"[^a-zA-Z ]","",text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

def prepocess_dataset(df):
    df['text']=df['text'].apply(clean_text)
    df=df.dropna()
    return df