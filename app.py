import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import preprocessing

st.header('COVID-19 Image Classifier')

def main():
    file_uploaded = st.file_uploader("Upload an Image of Chest X-Ray", type = ['.jpg', '.jpeg', '.png'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

@st.cache
def load_model():
    classifier_model = tf.keras.models.load_model('my_model_3.h5', compile=False)
    return classifier_model

def predict_class(image):
    classifier_model = load_model()
    with st.spinner('Sit tight, our model is about to predict...'):
        shape = (224, 224, 3)
        model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])
        test_image = image.resize((224, 224))
        test_image = preprocessing.image.img_to_array(test_image)
        test_image = test_image/255.0
        test_image = np.expand_dims(test_image, axis=0)
        class_names = ['COVID', 'Normal']
        predictions = model.predict(test_image)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()
        if predictions[0] < 0.5:
            return f"The image uploaded is COVID Infected with {round(100 - predictions[0][0] * 100, 2)} % confidence!!"
        else:
            st.balloons()
            return f"The image uploaded is Normal with {round(predictions[0][0] * 100, 2)} % confidence!!"


if __name__ == '__main__':
    main()
