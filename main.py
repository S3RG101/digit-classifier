import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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


canvas = st_canvas(
  fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
  stroke_width=20,
  stroke_color="rgba(0, 0, 0, 1)",
  background_color="rgba(0, 0, 0, 0)",
  update_streamlit=True,
  width=400,
  height=400,
  drawing_mode="freedraw",
  point_display_radius= 0,
  key="canvas",
)

def process_img(img):
  img.resize((28,28))
  img = img.convert('L')
  img = ImageOps.invert(img)
  img = np.array(img)
  img = img/255
  return img


if canvas.image_data is not None:
  drawing = Image.fromarray(np.uint8(canvas.image_data*255))
  drawing = process_img(drawing)
  st.image(drawing)

'''
def draw(filename='drawing.png', w=400, h=400, line_width=40):
  display(HTML(canvas_html % (w, h, line_width)))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  #return len(binary)

def show_img(img):
  plt.imshow(img, cmap=plt.cm.binary)
  plt.show()

increase_font()

drawing_test = drawing.reshape(1,784)
pred = logreg.predict(drawing_test)
print(f'Logistic Regression says {pred[0]}')

drawing_test = drawing.reshape(1,28,28)
pred = nn.predict(drawing_test)
max_idx = np.max(pred)
guess = np.argmax(pred)
print(f'Neural Networks says {guess} with probability {round(max_idx*100, 2)}%')

drawing_test = drawing.reshape(1,28,28)
pred = cnn.predict(drawing_test)
max_idx = np.max(pred)
guess = np.argmax(pred)
print(f'Convolutional Neural Networks says {guess} with probability {round(max_idx*100, 2)}%')

drawing_test = drawing.reshape(28,28)
drawing_test = np.pad(drawing_test, [(2,2), (2,2)], mode='constant')
drawing_test = drawing_test.reshape((1,32,32))
pred = lenet.predict(drawing_test)
max_idx = np.max(pred)
guess = np.argmax(pred)
print(f'LeNet-5 says {guess} with probability {round(max_idx*100, 2)}%')
'''
