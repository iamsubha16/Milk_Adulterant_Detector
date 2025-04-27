from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2

app = Flask(__name__, template_folder='templates')

model = tf.keras.models.load_model("model/model.keras")
class_names = ['Milk', 'Milk+Oil']

def predict_from_base64(base64_str):
    image_data = base64.b64decode(base64_str.split(',')[-1])
    image = Image.open(BytesIO(image_data)).convert('RGB')
    img = np.array(image.resize((256, 256)))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_milk = img[y:y+h, x:x+w]
    else:
        print(f"No contour found in. Skipping.")

    cropped_resized = cv2.resize(img, (256, 256))  # Resize again after cropping
    img_array = tf.keras.preprocessing.image.img_to_array(cropped_resized)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(round(100 * np.max(predictions[0]), 2))
    print(f"prediction Complete -- predicted_class{predicted_class}, confidence{confidence}")

    return predicted_class, confidence


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    try:
        predicted_class, confidence = predict_from_base64(data['image'])
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
