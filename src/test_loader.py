from src.data_loader import load_fake_news_dataset,save_dataset
from src.preprocessing import prepocess_dataset

df=load_fake_news_dataset()
df=prepocess_dataset(df)
print(df.head())
df.to_csv("data/processed/clean_news_dataset.csv",index=False)
# df=load_fake_news_dataset()
# save_dataset(df)
# print(df.head())
# print("Dataset size:",len(df))