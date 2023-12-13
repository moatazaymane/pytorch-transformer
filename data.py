import pandas as pd


df = pd.read_csv("en-fr.csv")

sample = df.tail(4000000)

print(df.head())

sample.to_csv("sample.csv", index=False)