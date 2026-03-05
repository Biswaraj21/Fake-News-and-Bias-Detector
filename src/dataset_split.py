import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset():
    df=pd.read_csv("data/processed/clean_news_dataset.csv")
    train, test=train_test_split(df,test_size=0.2,stratify=df["label"],random_state=42)
    train.to_csv("data/processed/train.csv",index=False)
    test.to_csv("data/processed/test.csv",index=False)
    
split_dataset()