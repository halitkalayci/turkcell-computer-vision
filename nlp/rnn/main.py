#RNN -> Recurrent Neural Network
# Modelleri kur.
# .summary() incele
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding,Input


model = Sequential()
model.add(Input(shape=(10,))) # 10 kelime SimpleRNN için müthiş, büyüdükçe problem!
model.add(Embedding( input_dim=10000, output_dim=64 )) # input_dim * output_dim = 64k
model.add(SimpleRNN(64)) # "32 nöronlu hafızalı beyin" (input_size+hidden_size) * hidden_size + hidden_size (64+32) * 32+32
model.add(Dense(1, activation="sigmoid")) # pozitif/negatif tahmini -> 32*1+1 => 33

model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])
model.summary()