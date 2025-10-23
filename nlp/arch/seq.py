import tensorflow as tf
from keras.layers import Input, LSTM, Dense
from keras.models import Model

# Girdi cümlesi (ör: 10 kelime, her kelime 64 boyutlu vektörlerle temsil edilmiş.)
# Merhaba
encoder_inputs = Input(shape=(None, 64))
encoder_lstm = LSTM(256, return_state=True) #return_state = "düşünce topu" h,c değerlerini verir.
_, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c] # [1,2,3] -> [0.15,0.35,0.50]

# Anlam vektörü alınır, Hello çıkartılır.
decoder_inputs = Input(shape=(None, 64))
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states) 
# tekil bi anlam üzerinden
# decode işlemi..

decoder_dense = Dense(10_000, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()