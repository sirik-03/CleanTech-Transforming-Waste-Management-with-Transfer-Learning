from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')  # Make sure 'model.h5' is in your project folder

# List of class names (change according to your model)
class_names = ['Biodegradable', 'Non-Biodegradable', 'Recyclable']

@app.route('/')
def index():
    return render_template('index.html')  # HTML form to upload image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save and process image
        image_path = os.path.join('static/uploads', file.filename)
        file.save(image_path)

        # Load and preprocess
        img = load_img(image_path, target_size=(224, 224))  # Adjust size to match your model
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template('result.html', 
                               prediction=predicted_class,
                               filename=file.filename)
    
    return redirect(url_for('index'))

# Optional: serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    app.run(debug=True)
