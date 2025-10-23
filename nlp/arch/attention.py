import tensorflow as tf
from keras.layers import Input, LSTM, Dense
from keras.layers import Attention, Concatenate
from keras.models import Model

# Girdi cümlesi (N adet kelime, her kelime 64 boyutlu vektör)
encoder_inputs = Input(shape=(None, 64))

#DEĞİŞTİ: Attention için encoder'ın tüm zaman adımlarını return etmesi lazım.
encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
# DEĞİŞTİ: Sadece stateleri değil, çıktıyı komple al çünkü tüm çıktılardaki "attention" hesaplanacak.
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
# AYNI
encoder_states = [state_h, state_c]

# AYNI
decoder_inputs = Input(shape=(None, 64))
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states) 
#

# EKLENDİ
attn_layer = Attention(name="attention")
context = attn_layer([decoder_outputs, encoder_outputs])
#

concat = Concatenate(axis=-1, name="concat_context")([decoder_outputs, context])

decoder_dense = Dense(10_000, activation="softmax")
decoder_outputs = decoder_dense(concat)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()