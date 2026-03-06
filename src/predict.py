from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("models/fake_news_model")

print("Model loaded successfully")