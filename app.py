from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import json

import pymongo

# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.applications.densenet import decode_predictions, preprocess_input

# from keras.preprocessing import image
from keras.utils import load_img, img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
print("successfully running")

from dotenv import load_dotenv
load_dotenv()
import cloudinary
import cloudinary.uploader
import cloudinary.api
config = cloudinary.config(secure=True)
print("****1. Set up and configure the SDK:****\nCredentials: ", config.cloud_name, config.api_key, "\n")
# from PIL import Image
# from io import BytesIO
import logging
# kode github
 
 # Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/test-model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')

def uploadImage(img):
#   image = Image.open(img.stream)
  # Upload the image and get its URL
  # ==============================
#   image_bytes = BytesIO()
#   image.save(image_bytes, format='JPEG')
#   image_bytes.seek(0)

  # Upload the image.
  # Set the asset's public ID and allow overwriting the asset with new versions
  cloudinary.uploader.upload(img)

  # Build the URL for the image and save it in the variable 'srcURL'
  srcURL = cloudinary.CloudinaryImage("quickstart_butterfly").build_url()

  # Log the image URL to the console. 
  # Copy this URL in a browser tab to generate the image on the fly.
  print("****2. Upload an image****\nDelivery URL: ", srcURL, "\n")
  
def model_predict(img_path, model):
    # img = image.load_img(img_path, target_size=(224, 224))
    print("pass modell_predict function")
    img = load_img(img_path, target_size=(100, 100, 3))

    # Preprocessing the image
    # x = image.img_to_array(img)
    img = img_to_array(img)
    img = img/255.0
    prediction_image = np.array(img)
    prediction_image = np.expand_dims(prediction_image, axis=0)
    # prediction_image = preprocess_input(prediction_image)

    
    preds = model.predict(prediction_image)
    print(preds)
    
    return preds


Name = ['Fresh', 'Spoiled']
N=[]
for i in range(len(Name)):
    N+=[i]
reverse_mapping=dict(zip(N,Name)) 
def mapper(value):
    return reverse_mapping[value]

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # uploadImage(f)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        response = cloudinary.uploader.upload(file_path)
        logging.info("response", response)
        
        value = np.argmax(preds)
        move_name = mapper(value)
        result = {"prediction": move_name}
        result["image_url"] = str(response["secure_url"])
        result["email"] = request.form.get("email")
        result = json.dumps(result)
        print(value)
        print("last steppppp")
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
