import tensorflow as tf
from keras import layers,models

inputs = layers.Input(shape=(None, 64))
#[
# [-0.46947439  0.54256004 -0.46341769 -0.46572975], 
# [-0.46947439  0.54256004 -0.46341769 -0.46572975]
#]

# Sırayla işlem yapılmadığı için (konumsal kodlama)
# (kelime_embedding + pozisyon)
# 0,  1,   2,    3
# Ben bugün okula gittim
# [-0.46947439  0.54256004 -0.46341769 -0.46572975, 1]

def add_positional_encoding(x):
    seq_len = tf.shape(x)[1]
    pos_encoding = tf.range(start=0, limit=seq_len, delta=1)
    pos_encoding = tf.cast(tf.expand_dims(pos_encoding, -1), tf.float32)
    return x + pos_encoding

x = layers.Lambda(add_positional_encoding)(inputs)
# Tüm encodingleri aldım, üzerine pozisyon bilgisini ekledim.

attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x,x)
x = layers.Add()([x, attn_out])
x = layers.LayerNormalization()(x)


ffn = layers.Dense(128, activation='relu')(x)
ffn = layers.Dense(64)(ffn)
x = layers.Add()([x,ffn])
x = layers.LayerNormalization()(x)

outputs = layers.Dense(10_000, activation="softmax")(x)

model = models.Model(inputs,outputs)
model.summary()

# Conversational - Forum