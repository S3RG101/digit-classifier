import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pickle

filename1 = "/digit-classifier/logreg.sav"
logreg = pickle.load(open(filename1, 'rb'))

filename2 = "/digit-classifier/neunet"
nn = keras.models.load_model(filename2)

filename3 = "/digit-classifier/cnn"
cnn = keras.models.load_model(filename3)

filename4 = "/digit-classifier/lenet"
lenet = keras.models.load_model(filename4)


from IPython.display import HTML, Image
from google.colab.output import eval_js
from base64 import b64decode

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import PIL

canvas_html = """
<canvas width=%d height=%d></canvas>
<button>Finish</button>
<script>
var canvas = document.querySelector('canvas')
var ctx = canvas.getContext('2d')
ctx.lineWidth = %d
ctx.fillStyle = "#ffffff";
ctx.fillRect(0, 0, canvas.width, canvas.height);
var button = document.querySelector('button')
var mouse = {x: 0, y: 0}
canvas.addEventListener('mousemove', function(e) {
  mouse.x = e.pageX - this.offsetLeft
  mouse.y = e.pageY - this.offsetTop
})
canvas.onmousedown = ()=>{
  ctx.beginPath()
  ctx.moveTo(mouse.x, mouse.y)
  canvas.addEventListener('mousemove', onPaint)
}
canvas.onmouseup = ()=>{
  canvas.removeEventListener('mousemove', onPaint)
}
var onPaint = ()=>{
  ctx.lineTo(mouse.x, mouse.y)
  ctx.stroke()
}
var data = new Promise(resolve=>{
  button.onclick = ()=>{
    resolve(canvas.toDataURL('image/png'))
  }
})
</script>
"""

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

def process_img(img):
  img = img.resize((28,28))

  img = img.convert('L')
  img = ImageOps.invert(img)
  img = np.array(img)

  # show_img(img)

  img = img/255

  return img

draw()

drawing = Image.open('drawing.png')

drawing = process_img(drawing)

#----------------------------------------
def increase_font():
  from IPython.display import Javascript
  display(Javascript('''
  for (rule of document.styleSheets[0].cssRules){
    if (rule.selectorText=='body') {
      rule.style.fontSize = '25px'
      break
    }
  }
  '''))

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
