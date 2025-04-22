from flask import Flask, request, render_template, url_for, send_from_directory , Response
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Improved Image Preprocessing
def preprocessing(img_path):
    img = cv2.imread(img_path)  # Read image in BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (32, 32))  # Resize to model input size
    img = cv2.equalizeHist(img)  # Histogram Equalization
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, 32, 32, 1)  # Ensure correct shape (Batch, Height, Width, Channels)
    return img

# ✅ Improved Prediction Function
def model_predict(img_path, model):
    img = preprocessing(img_path)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)  # Get the class with the highest probability
    confidence = np.max(predictions)  # Confidence score
    class_name = getClassName(classIndex)
    
    print(f"Predicted Class: {class_name}, Confidence: {confidence:.2f}")
    
    return f"{class_name} (Confidence: {confidence:.2f})"

# ✅ Class Name Mapping (Same as Before)
def getClassName(classNo):
    class_names = {
        0: 'Give way', 1: 'No entry', 2: 'One-way traffic', 3: 'One-way traffic',
        4: 'No vehicles in both directions', 5: 'No entry for cycles', 
        6: 'No entry for goods vehicles', 7: 'No entry for pedestrians',
        8: 'No entry for bullock carts', 9: 'No entry for hand carts',
        10: 'No entry for motor vehicles', 11: 'Height limit', 12: 'Weight limit',
        13: 'Axle weight limit', 14: 'Length limit', 15: 'No left turn', 
        16: 'No right turn', 17: 'No overtaking', 18: 'Maximum speed limit (90 km/h)',
        19: 'Maximum speed limit (110 km/h)', 20: 'Horn prohibited', 21: 'No parking',
        22: 'No stopping', 23: 'Turn left', 24: 'Turn right', 25: 'Steep descent',
        26: 'Steep ascent', 27: 'Narrow road', 28: 'Narrow bridge', 29: 'Unprotected quay',
        30: 'Road hump', 31: 'Dip', 32: 'Loose gravel', 33: 'Falling rocks', 
        34: 'Cattle', 35: 'Crossroads', 36: 'Side road junction', 37: 'Side road junction',
        38: 'Oblique side road junction', 39: 'Oblique side road junction', 40: 'T-junction',
        41: 'Y-junction', 42: 'Staggered side road junction', 43: 'Staggered side road junction',
        44: 'Roundabout', 45: 'Guarded level crossing ahead', 46: 'Unguarded level crossing ahead',
        47: 'Level crossing countdown marker', 48: 'Level crossing countdown marker',
        49: 'Level crossing countdown marker', 50: 'Level crossing countdown marker',
        51: 'Parking', 52: 'Bus stop', 53: 'First aid post', 54: 'Telephone',
        55: 'Filling station', 56: 'Hotel', 57: 'Restaurant', 58: 'Refreshments'
    }
    return class_names.get(classNo, "Unknown")

# ✅ Flask Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    f = request.files['file']
    if f.filename == '':
        return "No file selected", 400

    # Save file
    filename = secure_filename(f.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(file_path)

    # Make prediction
    preds = model_predict(file_path, model)

    # Return results in an HTML page
    return render_template("result.html", prediction=preds)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(port=5001, debug=True)
