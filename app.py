import gradio as gr
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Load the pre-trained model at the start
model = load_model('model/best_cat_dog_classifier.keras')

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Prediction function
def predict_image(img):
    # Save the image temporarily
    img.save("temp_image.png")
    img_path = "temp_image.png"

    # Preprocess and predict
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    confidence_threshold = 0.5

    if prediction >= confidence_threshold:
        result = "dog"
        confidence = float(prediction) * 100
    else:
        result = "cat"
        confidence = float(1 - prediction) * 100

    # Clean up
    os.remove(img_path)
    return {"result": result, "confidence": round(confidence, 2)}

# Define Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.inputs.Image(type="pil"),
    outputs="json",
    title="Cat vs Dog Classifier",
    description="Upload an image to classify it as a cat or dog."
)

# Launch the interface
interface.launch()


