import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


tokenizer=BertTokenizer.from_pretrained("models/fake_news_model")
model=BertForSequenceClassification.from_pretrained("models/fake_news_model")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


df=pd.read_csv("data/processed/test.csv")
df=df.dropna(subset=["text"])
df["text"]=df["text"].astype(str)

texts=df["text"].tolist()
labels=df["label"].tolist()


class TestDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings=tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256
        )

        self.labels=labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


dataset=TestDataset(texts, labels)
loader=DataLoader(dataset, batch_size=8)


predictions=[]
true_labels=[]

with torch.no_grad():
    for batch in loader:
        input_ids=batch["input_ids"].to(device)
        attention_mask=batch["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch["labels"].numpy())


accuracy=accuracy_score(true_labels, predictions)
precision=precision_score(true_labels, predictions)
recall=recall_score(true_labels, predictions)
f1=f1_score(true_labels, predictions)
cm=confusion_matrix(true_labels, predictions)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("Confusion Matrix:")
print(cm)