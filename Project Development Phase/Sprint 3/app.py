import os
import pybase64
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

from tensorflow import keras
from keras.models import load_model
from flask import Flask, render_template, request

model = load_model('models/mnistCNN.h5')

app = Flask(__name__)

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
        im2arr = im2arr.reshape(1, 28, 28, 1)
        prediction = model.predict(im2arr)
        y_pred = np.argmax(prediction)

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
