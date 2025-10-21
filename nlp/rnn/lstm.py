from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding,Input

from nlp.rnn.dataset import X_train, max_features, max_len, y_train


model = Sequential()
model.add(Input(shape=(max_len,))) # 10 kelime SimpleRNN için müthiş, büyüdükçe problem!
model.add(Embedding( input_dim=max_features, output_dim=64 )) # input_dim * output_dim = 64k
model.add(LSTM(64)) # "32 nöronlu LSTM" -> (input_size+hidden_size + 1) * 4 * hidden_size => (64+32+1) * 4 * 32 => 
model.add(Dense(1, activation="sigmoid")) # pozitif/negatif tahmini -> 32*1+1 => 33

model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

lstm_model = model.fit(X_train, y_train, epochs=8, batch_size=64,validation_split=0.2)
model.save("lstm_model.keras")
# 4 Gates 
# Forget,Input,Cell State,Output