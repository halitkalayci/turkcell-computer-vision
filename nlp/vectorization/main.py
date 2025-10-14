# ML -> Machine Learning yalnızca sayılarla işlem yapabilir.

# Numerik Veri

# Her kelimeye bir sayı verelim. -> "ama" 0, "hiç" 1

# Bag-Of-Words
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "kahve sıcak ve güzel kahve", #mutlu
    "çay da sıcak ve lezzetli", #üzgün
    "kahve ve çay favori içeceklerim" #mutlu
]

print("1. adım - CountVectorizer")
vectorizer = CountVectorizer()
vectorizer.fit(corpus)
vocabulary = vectorizer.vocabulary_ # O kelimeye atanan sayı.
print("Kelime listesi: ", vocabulary)
print("-"*50)

vector = vectorizer.transform(corpus)
print(vector.toarray())


column_names = vectorizer.get_feature_names_out()

import pandas as pd

df = pd.DataFrame(vector.toarray(), columns=column_names)
print(df)