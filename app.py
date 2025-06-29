from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model("best_model_out.keras")
print("✅ Model loaded successfully")

# Fabric class labels
class_labels = [
    "argyle", "camouflage", "checked", "dot", "floral", "geometric",
    "gradient", "graphic", "houndstooth", "leopard", "lettering",
    "muji", "paisley", "snake_skin", "snow_flake", "stripe", "tropical",
    "zebra", "zigzag"
]

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file uploaded", 400

    try:
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]

        # Ensure prediction has correct size
        if len(prediction) != len(class_labels):
            return "Model output size doesn't match number of class labels.", 500

        # Get top 2 predictions
        top_indices = prediction.argsort()[-2:][::-1]
        top_labels = [(class_labels[i], round(prediction[i] * 100, 2)) for i in top_indices]

        # Get all class probabilities
        confidences = {class_labels[i]: f"{round(prediction[i]*100, 2)}%" for i in range(len(class_labels))}

        return render_template(
            'result.html',
            label=top_labels[0][0],
            confidence=top_labels[0][1],
            second_label=top_labels[1][0],
            second_confidence=top_labels[1][1],
            confidences=confidences,
            image_path=filepath
        )
    
    except Exception as e:
        return f"❌ Internal error during prediction: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
