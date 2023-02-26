from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import json

# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.applications.densenet import decode_predictions, preprocess_input

# from keras.preprocessing import image
from keras.utils import load_img, img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

print("successfully running")

# kode github
 
 # Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/test-model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

# default prediction section
# def model_predict(img_path, model):
#     # img = image.load_img(img_path, target_size=(224, 224))
#     img = load_img(img_path, target_size=(224, 224))

#     # Preprocessing the image
#     # x = image.img_to_array(img)
#     x = img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')

#     preds = model.predict(x)
#     return preds

# coba coba prediction section
# Name=[]
# N=[]
# reverse_mapping=dict(zip(N,Name)) 
# def mapper(value):
#     return reverse_mapping[value]

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
    # value = np.argmax(preds)
    # move_name = mapper(value)
    # print(value)
    # res = format(move_name)
    return preds


# chatGPT code 
# def model_predict(img_path, model):
#     # Load and preprocess the image
#     img = load_img(img_path, target_size=(100, 100))
#     x = img_to_array(img)
#     x = preprocess_input(x)

#     # Make prediction
#     preds = model.predict(x)

#     # Decode the predictions
#     pred_class = decode_predictions(preds, top=5)[0]

#     # Format the predictions as a string
#     result = ''
#     for pred in pred_class:
#         result += f'{pred[1]}: {pred[2]:.4f}\n'

#     return result
# end chatGPT code

# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')
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

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # solusi chatGPT rada ngawur
        # pred_array = np.zeros((1, 1000))
        # # Set the relevant elements of the array to the output of the model
        # for i, p in enumerate(preds[0]):
        #     pred_array[0][i] = p
        # end solusi chatGPT rada ngawur
        
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=3)[0]   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        value = np.argmax(preds)
        move_name = mapper(value)
        result = "predictions is {}.".format(move_name)
        result = json.dumps(result)
        print(value)
        print("last steppppp")
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
