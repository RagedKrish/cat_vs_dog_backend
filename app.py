# app.py
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=['https://cat-vs-dog-sigma.vercel.app/'])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

model = load_model('best_cat_dog_classifier.keras')  # Load the pre-trained model
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import os

# Set up the uploads directory
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Adjust for model input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img /= 255.0
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = preprocess_image(file_path)
        prediction = model.predict(img)
        confidence_threshold = 0.5

        if prediction >= confidence_threshold:
            result = "dog"
            confidence = (float(prediction))*100
        else:
            result = "cat"
            confidence = (float(1-prediction))*100

        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

        return jsonify({'result': result, 'confidence': round(confidence, 2)})
    return jsonify({'error': 'Failed to process the image'}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
