import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# TODO: Çalışır hale getir.

def load_model(model_path="best_model.keras"):
    """Loads the Keras model from the specified path."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(img_path):
    """Loads and preprocesses an image for the MNIST model."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def grad_cam(img, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap /= max_val
    heatmap = heatmap.numpy()

    return heatmap, pred_index.numpy()


def show_grad_cam(img_path, model, last_conv_layer_name="conv2d_1"):
    """
    Loads an image, generates, and displays the Grad-CAM heatmap.
    """
    # A dummy pass to build the model
    prediction = model(np.zeros((1, 28, 28, 1)))
    print(prediction)

    img = preprocess_image(img_path)
    
    heatmap, predicted_class = grad_cam(img, model, last_conv_layer_name)

    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (28, 28))


    heatmap = cv2.resize(heatmap, (28, 28))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM (Predicted: {predicted_class})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model = load_model("best_model.keras")
    if model:
        # You should have 'image.png' or another image file in the cnn directory
        # to test this script.
        image_path = "image2.png" 
        try:
            show_grad_cam(image_path, model)
        except Exception as e:
            print(f"Could not process {image_path}. Make sure the file exists. Error: {e}")
