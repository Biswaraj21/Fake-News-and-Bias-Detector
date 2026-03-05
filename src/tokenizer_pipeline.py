import pandas as pd
import torch
from transformers import BertTokenizer

tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_dataset():
    df=pd.read_csv("data/processed/train.csv")
    df=df.dropna(subset=["text"])
    df["text"]=df["text"].astype(str)
    tokens=tokenizer(df["text"].tolist(),padding=True,truncation=True,max_length=256,return_tensors='pt')
    labels = torch.tensor(df["label"].values)
    return tokens, labels

if __name__=="__main__":
    tokens,labels=tokenize_dataset()
    print("Input IDs shape:", tokens["input_ids"].shape)
    print("Attention mask shape:", tokens["attention_mask"].shape)
    print("Labels shape:", labels.shape)