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


print("-"*50)
print("2. adım - TF-IDF")

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer2 = TfidfVectorizer()

# fit => Corpusu analiz eder. Kelimelerden unique bir sözlük çıakrt.
# fit => Her kelimenin IDF'ini hesaplar.
# transform => Her bir cümleyi, her kelime için TF*IDF skorunu hesaplayarak vectorize eder.
tfidf_matrix = vectorizer2.fit_transform(corpus)
vocabulary2 = vectorizer2.vocabulary_
print("Kelime listesi: ", vocabulary2)

idf_scores = vectorizer2.idf_
print("IDF skorları: ", idf_scores)

df_idf = pd.DataFrame(idf_scores, index=vectorizer2.get_feature_names_out(), columns=["IDF"])
print(df_idf)

print("-"*50)

df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer2.get_feature_names_out())
print(df_tfidf)

corpus = [
    "kahve sıcak ve güzel kahve", #mutlu
    "çay da sıcak ve lezzetli", #üzgün
    "kahve ve çay favori içeceklerim" #mutlu
]

# güzel
# 1 belgede geçiyor df = 1
# IDF = log((1+3) / (1+1)) + 1 = log(4/2) + 1 = log(2) + 1

# ve
# 3 belgede geçiyor
# IDF = log((1+3) / (1+3)) + 1 = log(4/4) + 1 = log(1) + 1