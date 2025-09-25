import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import cv2

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

def show_images():
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


def define_and_train_model():
    # Early Stopping => 27. epochta accuracy %99 du, 28,29,30'da azalmaya başladı (patience sayısı kadar) azalmayı göz ardı et.
    # geçtiği an durdur. 

    callbacks = [
        # düşüşten 3. epoch geçtikten sonra eğitimi durdur.
        tf.keras.callbacks.EarlyStopping(patience=6, monitor="val_accuracy", restore_best_weights=True),
        # tüm epochlar içinden en iyi skora sahip olanı seç.
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
        # learning rate oranını otomatik düşür.
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=0.0001)
    ]

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)), # 32 dedektif bilgi çıkarımı yap.
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)), #Bilgiyi özetler.
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"), # 64 dedektif bilgi çıkarımı yap.
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)), #Bilgiyi özetler.
        tf.keras.layers.Flatten(), #Bir boyutlu yapıya çevir.
        tf.keras.layers.Dense(units=128, activation="relu"), # 128 nöronluk karar mekanizması.
        tf.keras.layers.Dense(units=10, activation="softmax") # 10 nöronluk karar mekanizması.
    ])
    model.summary()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=25, callbacks=callbacks)
    model.save("mnist_model.keras")

#define_and_train_model()

def load_model():
    model = tf.keras.models.load_model("best_model.keras")
    return model

def confusion_matrix():
    model = load_model()
    y_pred = np.argmax(model.predict(X_test), axis=1)
    # precision -> kesinlik (pozitif öngörü doğruluğu) 0 dediklerimin % kaçı gerçekten 0?
    # recall -> duyarlılık (yakalama oranı) 1 için %100 -> Test setindeki tüm 1leri yakalamışım.
    # f1-score -> precision ve recall'un ortalaması
    # support -> Test setindeki sınıf sayısı
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,10))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.show()

def predict(img_path):
    model = load_model()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)

    img = img/255.0
    img = img.reshape(-1,28,28,1)

    probs = model.predict(img)[0]
    pred = np.argmax(probs)
    return pred,probs

def predict_x_images(number):
    model = load_model()
    indices = np.random.choice(len(X_test), number, replace=False)

    y_probs = model.predict(X_test[indices])
    y_pred = np.argmax(y_probs, axis=1)
    y_true = y_test[indices]

    plt.figure(figsize=(15,10))

    for i, indx in enumerate(indices):
        plt.subplot(5,6, i+1)
        plt.imshow(X_test[indx].reshape(28,28), cmap="gray")
        plt.title(f"Tahmin: {y_pred[i]}, Gerçek: {y_true[i]}")
        plt.axis("off")

    plt.show()
    
#confusion_matrix()
#pred, probs = predict("image2.png")
#print(f"Tahmin: {pred}")
#print(f"Olasılıklar: {probs}")
predict_x_images(20)
