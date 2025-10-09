import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

turkish_stop_words = stopwords.words("turkish")

sentence = "Dr. Ömer, İstanbul'a saat 14:30'da vardı. Dr. Ömer'in yazdığı bu makale yapay zeka ve onun geleceği hakkında bilgiler içermektedir ama ben henüz okumadım."

tokens = word_tokenize(sentence, language="turkish")

cleaned_tokens = [token for token in tokens if token.lower() not in turkish_stop_words]

print(tokens)
print("********** Cleaned ***********")
print(cleaned_tokens)

# TODO: nltk stemmer Türkçe
# Stemming
from nltk.stem.snowball import SnowballStemmer
tr_stemmer = SnowballStemmer("turkish")

words = ["kitaplar","gözlükçü","koşuyordu","güzelleştiriyor","gidiyorum","kitapçı","gözlük"]

for word in words:
    print(f"{word} -> {tr_stemmer.stem(word)}")
#


# The goal of the task (Görevin amacı)
# Duygu analizi -> noktalama işaretleri önemli, dolayısıyla noktalama işarelteri tek başına ayrı birer eleman olmalı.
# Metin Sınıflandırma -> Kelime kökleri

# The structure of language (Dilin Yapısı)
# İngilizce ->  Book, books, my books, in my books
# Türkçe (Bitişken) -> kitap, kitaplar, kitaplarım, kitaplarımdaki, 

# The structure of domain (Verinin ve domainin yapısı)
# Sosyal Medya -> #yapayzeka #yapay #zeka slm nbr -> TweetTokenizer
# Tıbbi veya Hukuki metinler -> 


# The arch. of model (Kullanılacak modelin mimarisi)
# Klasik Modeller (Bag-of-Words, Tf-IDF) 
# Transformer Modelleri (BERT, GPT)