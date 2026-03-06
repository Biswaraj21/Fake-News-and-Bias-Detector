import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,BertForSequenceClassification
from torch.optim import AdamW

tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")

class FakeNewsDataset(Dataset):
    def __init__(self,texts,labels):
        self.encodings=tokenizer(texts,truncation=True,padding=True,max_length=256)
        self.labels=labels
    
    def __getitem__(self,idx):
        item={key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item["labels"]=torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
df=pd.read_csv('data/processed/train.csv')
df=df.dropna(subset=["text"])
df["text"]=df["text"].astype(str)
texts=df["text"].tolist()
labels=df['label'].tolist()
dataset=FakeNewsDataset(texts,labels)
loader=DataLoader(dataset,batch_size=8,shuffle=True)
model=BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer=AdamW(model.parameters(),lr=2e-5)

epochs=3
model.train()
for epoch in range(epochs):
    total_loss=0
    for batch in loader:
        optimizer.zero_grad()
        input_ids=batch["input_ids"].to(device)
        attention_mask=batch["attention_mask"].to(device)
        labels=batch["labels"].to(device)
        outputs=model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        loss=outputs.loss
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1} Loss:",total_loss)

model.save_pretrained("models/fake_news_model")
tokenizer.save_pretrained("models/fake_news_model")