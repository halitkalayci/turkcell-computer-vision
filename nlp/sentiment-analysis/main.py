import html
import string
import pandas as pd
import re

df = pd.read_csv("data/dataset.csv")


df["label"] = df["sentiment"].apply(lambda x: 1 if x == 'positive' else 0)


html_tag = re.compile(r"<.*?>")
url_path = re.compile(r"http\S+|www\.\S+")
mention = re.compile(r"@\S+")
multi_spaces = re.compile(r"\s+")
punct_table = str.maketrans("", "", string.punctuation)

def clean_text(text: str) -> str: 
    text = html.unescape(text)
    text = re.sub(html_tag, " ", text)
    text = re.sub(url_path, " ", text)
    text = re.sub(mention, " ", text)
    text = re.sub(multi_spaces, " ", text)
    text = text.translate(punct_table)
    return text.strip().lower()

df["text"] = df["review"].apply(clean_text)
print(df.head())

#df.to_csv("data/cleaned_dataset.csv", index=False)