# streamlit_app.py

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load the trained model
model = keras.models.load_model('path_to_your_trained_model')

# Load MNIST test dataset
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test / 255.0
x_test_flattened = x_test.reshape(len(x_test), 28 * 28)

# Function to preprocess input image
def preprocess_image(image):
    # Ensure that the image is 28x28
    image = tf.image.resize(image, (28, 28))
    image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale if needed
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, (1, 784)).numpy() / 255.0
    return image

# Streamlit app
st.title("Digit Classification with Streamlit and Keras")

uploaded_file = st.file_uploader("Choose an image of a digit...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = tf.image.decode_image(uploaded_file.read(), channels=3)  # Adjust channels based on your image
    st.image(image.numpy().squeeze(), caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image and make predictions
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_label = np.argmax(predictions[0])

    st.write(f"Predicted Digit: {predicted_label}")
    st.write("Prediction Probabilities:")
    st.bar_chart(predictions[0])

# Display evaluation results on the test set
st.subheader("Model Evaluation on Test Set")

# Convert labels to integers for evaluation
y_test = y_test.astype(int)

test_loss, test_accuracy = model.evaluate(x_test_flattened, y_test)
st.write(f"Test Loss: {test_loss}")
st.write(f"Test Accuracy: {test_accuracy}")
