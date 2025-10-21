import html
import string
import pandas as pd
import re
import joblib

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

# --- Split ----
from sklearn.model_selection import train_test_split
# Stratifed Sampling -> Sınıf dağılımını koruyarak eşit sayıda örnek al.

X_train, X_temp, y_train, y_temp = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ---- Split ----

## Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        max_features=100_000,
        min_df=2, # çok nadir kelimeleri atla min_df
        sublinear_tf=True, # harika harika harika harika film
        # Bu müthiş bir film
        # Bu muthis bir film
        strip_accents="unicode", # unicode karakterleri strip et (ş,ö,ü) -> (s,o,u)
    )),
    ("model", LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=100,
        class_weight=None,
    ))
])

# 1-100 -> "facia" -20
# 1-100 -> "güzel" +5
# [513,510,235,162]
# -20
# +5


# GridSearchCV
model = pipe.fit(X_train, y_train)

print(model.score(X_valid, y_valid))
joblib.dump(model, "models/sentiment_analysis_model.joblib")
##