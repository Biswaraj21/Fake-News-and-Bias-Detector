import streamlit as st
import torch
from transformers import BertTokenizer,BertForSequenceClassification

@st.cache_resource
def load_model():
    model=BertForSequenceClassification.from_pretrained("models/fake_news_model",attn_implementation='eager')
    tokeniser=BertTokenizer.from_pretrained("models/fake_news_model")
    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model,tokeniser,device

model,tokeniser,device=load_model()
st.title("Fake News Detection System")
st.write("Paste a news article below and the AI model will determine wheather it is likely fake or real")
text=st.text_area("Enter News Article",height=200)

def predict(text):
    inputs=tokeniser(text,return_tensors='pt',truncation=True,padding=True,max_length=256)
    inputs={k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs=model(**inputs)
        logits=outputs.logits
        prob=torch.softmax(logits,dim=1)
        prediction=torch.argmax(logits,dim=1).item()
        confidence=prob.max().item()
    return prediction,confidence

if st.button("Analyse Article"):
    if text.strip()=="":
        st.warning("Please enter a news article.")
    else:
        pred,conf=predict(text)
        if pred==1:
            st.error(f"Fake News (Confidence: {conf:.2f})")
        else:
            st.success(f"Real News (Confidence: {conf:.2f})")
            
if st.button("Load Example Article"):
    text="""The government announced a new economic reform package aimed at improving small business growth. 
According to the finance ministry, the policy includes tax reductions for startups and expanded 
credit access for small enterprises. Economists say the move could stimulate investment and job 
creation over the next few years, although some experts warn that the impact will depend on how 
effectively the policies are implemented."""