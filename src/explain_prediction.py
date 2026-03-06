import torch
from transformers import BertTokenizer,BertForSequenceClassification

model=BertForSequenceClassification.from_pretrained('models/fake_news_model',attn_implementation="eager")
tokeniser=BertTokenizer.from_pretrained('models/fake_news_model')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

text = """
Government secretly approves controversial policy affecting millions of citizens.
Experts warn that the decision could damage economic stability.
"""
inputs=tokeniser(text,return_tensors='pt',truncation=True,padding=True,max_length=256)
inputs={key:val.to(device) for key,val in inputs.items()}
outputs=model(**inputs,output_attentions=True)
logits=outputs.logits
attention=outputs.attentions

prediction=torch.argmax(logits,dim=1).item()
if prediction==1:
    print("Prediction: Fake News")
else:
    print("Prediction: Real News")
last_attention=attention[-1]
scores=last_attention.mean(dim=1).squeeze()
tokens=tokeniser.convert_ids_to_tokens(inputs['input_ids'][0])
for token,score in zip(tokens,scores.mean(dim=0)):
    print(token,float(score))