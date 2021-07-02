from fastapi import FastAPI, Request, UploadFile, File, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from keras.models import load_model
from keras.preprocessing import image 
from keras.applications.inception_v3 import preprocess_input
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = load_model('model.h5')
model.make_predict_function()

label = {
    0 : 'Covid',
    1 : 'Normal',
    2 : 'Viral Pneumonia'
}

def predict_label(img_path):
  img = image.load_img(img_path, target_size=(299,299))
  img = image.img_to_array(img)
  img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
  img = preprocess_input(img)

  y_hat = model.predict(img)
  return label[np.argmax(y_hat)]

app.get('/', response_class = HTMLResponse)
def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request" : request})

@app.post('/predict')
def get_predict(file: UploadFile = File(...)):
  img_path = file.filename
  
  predict = predict_label(img_path)
  return predict
  #return templates.TemplateResponse('index.html', prediction = predict, img_path = img_path)