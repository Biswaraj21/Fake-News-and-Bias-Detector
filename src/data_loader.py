import pandas as pd

def load_fake_news_dataset():
    fake=pd.read_csv("data/raw/Fake.csv")
    true=pd.read_csv("data/raw/True.csv")
    
    fake["label"]=1
    true["label"]=0
    
    df=pd.concat([fake,true],ignore_index=True)
    df=df[["text",'label']]
    return df

def save_dataset(df):
    df.to_csv("data/processed/news_dataset.csv",index=False)
    