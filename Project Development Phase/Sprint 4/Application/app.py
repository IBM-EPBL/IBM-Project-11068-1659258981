import os
import pybase64
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

from tensorflow import keras
from keras.models import load_model
from flask import Flask, render_template, request

app = Flask(__name__)

API_KEY = "" # Removed API Key before pushing to github
API_ENDPOINT = "https://us-south.ml.cloud.ibm.com/ml/v4/deployments/680e8d3b-1e5b-40e9-b639-89cf48bcdf6e/predictions?version=2022-11-26"

token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]
header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/upload')
def upload_file2():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def upload_image_file():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(request.files['file'].stream).convert('L')
        img = img.resize((28, 28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(28, 28, 1)
        
        payload_scoring = {"input_data": [{"fields": [], "values": [im2arr.tolist()]}]}
        response_scoring = requests.post(
            API_ENDPOINT,
            json=payload_scoring, 
            headers={'Authorization': 'Bearer ' + mltoken}
        )

        response = response_scoring.json()
        print(response)
        prediction = response['predictions'][0]['values'][0]
        y_pred = response['predictions'][0]['values'][0][1]

        prediction_percentage = str(round(max(prediction[0])*100, 2))+"%"
        filename = file.filename
        path = os.path.join("static/images", filename)
        img = Image.open(file.stream)
        file.save(path)

        if filename.endswith('jpg') or filename.endswith('jpeg'):
            with BytesIO() as buf:
                img.save(buf, 'jpeg')
                image_bytes = buf.getvalue()
            encoded_string = pybase64.b64encode(image_bytes).decode()
            encoded_string = "data:image/jpeg;base64,"+encoded_string
        if filename.endswith('png'):
            with BytesIO() as buf:
                img.save(buf, 'png')
                image_bytes = buf.getvalue()
            encoded_string = pybase64.b64encode(image_bytes).decode()
            encoded_string = "data:image/png;base64,"+encoded_string
        os.remove(path)

        if 0 <= y_pred <= 9:
            return render_template("result.html", digit=y_pred, user_image=encoded_string, percentage=prediction_percentage, showcase=str(y_pred))
        else:
            return render_template("result.html", digit="No digit found.", user_image=encoded_string, percentage=prediction_percentage)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
