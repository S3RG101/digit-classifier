import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow import keras
import pickle

from PIL import Image, ImageOps
import PIL
from base64 import b64decode

filename1 = "./logreg.sav"
logreg = pickle.load(open(filename1, 'rb'))

filename2 = "./neunet"
nn = keras.models.load_model(filename2)

filename3 = "./cnn"
cnn = keras.models.load_model(filename3)

filename4 = "./lenet"
lenet = keras.models.load_model(filename4)

st.title('Digit Recognition')

col1, col2 = st.columns(2)

with col1:
  canvas = st_canvas(
    fill_color="#ffffff",
    stroke_width=30,
    stroke_color="rgba(0, 0, 0, 1)",
    background_color="rgba(0, 0, 0, 0)",
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas")

if canvas.image_data is not None:
  drawing = Image.fromarray(canvas.image_data.astype('uint8')).convert('RGBA')
  resized_drawing = drawing.resize((28, 28))
  st.image(resized_drawing, caption="Resized Image (28x28)", width=400)

  resized_drawing.load()
  background = Image.new("RGB", resized_drawing.size, (255, 255, 255))
  background.paste(resized_drawing, mask = resized_drawing.split()[3])

  pixels = np.array(ImageOps.invert(background.convert('L'))) / 255.0
  st.write("Pixel Data:")
  st.write(pixels)
  st.write(pixels.shape)

drawing_test = pixels.reshape(1,784)
logregpred = logreg.predict(drawing_test)

drawing_test = pixels.reshape(1,28,28)
pred = nn.predict(drawing_test)
nnmax_idx = np.max(pred)
nnpred = np.argmax(pred)

drawing_test = pixels.reshape(1,28,28)
pred = cnn.predict(drawing_test)
cnnmax_idx = np.max(pred)
cnnpred = np.argmax(pred)

drawing_test = pixels.reshape(28,28)
drawing_test = np.pad(drawing_test, [(2,2), (2,2)], mode='constant')
drawing_test = drawing_test.reshape((1,32,32))
pred = lenet.predict(drawing_test)
lenetmax_idx = np.max(pred)
lenetpred = np.argmax(pred)

with col2:
  st.write(f'Logistic Regression says {logregpred[0]}')
  st.write(f'Neural Networks says {nnpred} with probability {round(nnmax_idx*100, 2)}%')
  st.write(f'Convolutional Neural Networks says {cnnpred} with probability {round(cnnmax_idx*100, 2)}%')
  st.write(f'LeNet-5 says {lenetpred} with probability {round(lenetmax_idx*100, 2)}%')
  
