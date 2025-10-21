from keras.datasets import imdb
from keras.preprocessing import sequence
import random

max_features= 10_000
max_len = 200

(X_train, y_train), (X_test,y_test) = imdb.load_data(num_words=max_features)

X_train = sequence.pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = sequence.pad_sequences(X_test,maxlen=max_len, padding="post")

print("Eğitim veri seti boyutu:", X_train.shape)
print("Test veri seti boyutu:", X_test.shape)

#print(f"{len(X_train[20])}")
#print(f"Rastgele bir yorum {X_train[60]}  ....")