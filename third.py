import pandas as pd
dataset=pd.read_csv("online.csv")
dataset.isnull()
dataset.isnull.head(10)
dataset.isnull().sum()
modifieddataset=dataset.fillna("")
modifieddataset.isnull().sum()
dataset=dataset.dropna()