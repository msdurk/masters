import pandas as pd

df = pd.read_csv("SpamAssasin.csv")

df = df[["sender", "subject", "body", "label"]]
df["label"] = df["label"].map({1: "Phishing", 0: "Valid"})

print(df.head())
df.to_csv("SpamAssasin_cleaned.csv", index=False)