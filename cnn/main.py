import tensorflow as tf
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

plt.figure(figsize=(10,10))

for i in range(10):
    plt.subplot(10,10, i+1)
    plt.imshow(X_train[i].reshape(28,28), cmap="gray")
    plt.axis("off")
plt.suptitle("İLK 10 Görüntü")
plt.show()

# Normalizasyon -> RGB değerleri 0-255 aralığından 0-1 aralığına taşı.
X_train = X_train / 255
X_test = X_test / 255
#
# örnek_sayısı,genişlik,yükseklik,kanal sayısı
X_train = X_train.reshape(-1,28,28,1) 
X_test = X_test.reshape(-1,28,28,1)


# Sequential -> Katmanları sıralı ilerleyen bir NN

# Aktivasyon Katmanı => Öğrendiğin bilgileri bana göster, ben hangilerini bu kapıdan geçirebileceğinin kararını vereceğim.
# ReLU -> Rectified Linear Unit -> Yalnızca Pozitifler geçer.
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)), # 32 dedektif bilgi çıkarımı yap.
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)), #Bilgiyi özetler.
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"), # 64 dedektif bilgi çıkarımı yap.
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)), #Bilgiyi özetler.
    tf.keras.layers.Flatten(), #Bir boyutlu yapıya çevir.
    tf.keras.layers.Dense(units=128, activation="relu"), # 128 nöronluk karar mekanizması.
    tf.keras.layers.Dense(units=10, activation="softmax") # 10 nöronluk karar mekanizması.
])
# 32 filtre
# 1.filter (dedektif) => fotoğrafı tamamen tara sadece dikey çizgileri bul
# 2.filter (dedektif) => fotoğrafı tamamen tara sadece yatay çizgileri bul
# 3.filter (dedektif) => fotoğrafı tamamen tara sadece çapraz çizgileri bul
# 32 adet Feature Map
model.summary()
# Eğitime hazır hale getirmek.
# Optimizer -> PERŞEMBE
# Loss -> PERŞEMBE
# Metrics -> PERŞEMBE
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=5, validation_split=0.2, batch_size=25)

model.save("mnist_model.keras")