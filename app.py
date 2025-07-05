import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from keras.initializers import Orthogonal
import numpy as np
import pickle
import os
import uuid

# Load retrained model with Orthogonal initializer support
model = load_model('model.keras', compile=False, custom_objects={
    'Orthogonal': Orthogonal
})

# Load tokenizer
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# Set your model's max caption length
max_length = 35

# Load VGG16 for image feature extraction
base_model = VGG16()
vgg_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

# Extract features from image
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = vgg_model.predict(image, verbose=0)
    return features

# Generate caption from features
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# Streamlit UI
st.set_page_config(page_title="🖼️ Image Caption Generator", layout="centered")
st.title("🖼️ Image Caption Generator")
st.write("Upload an image to generate a caption using your retrained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", str(uuid.uuid4()) + "_" + uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        photo = extract_features(file_path)
        caption = generate_caption(model, tokenizer, photo, max_length)

    st.success("Caption Generated:")
    st.markdown(f"**{caption}**")

    os.remove(file_path)
