# predict.py
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Definisikan parameter yang sama dengan saat training
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path, saved_model):
    """Fungsi untuk memprediksi satu gambar."""
    try:
        img = tf.keras.utils.load_img(
            image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = saved_model.predict(img_array)
        score = predictions[0][0]

        # Karena: Fake = 0, Real = 1
        if score < 0.5:
            result = 'FAKE'
            confidence = 100 * (1 - score)
        else:
            result = 'REAL'
            confidence = 100 * score

        return {
            'success': True,
            'result': result,
            'confidence': float(confidence)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# Muat model yang telah dilatih
try:
    loaded_model = tf.keras.models.load_model('deepfake_detector_model2.h5')
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'})
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_image(filepath, loaded_model)
        
        # Add image path to result if successful
        if result['success']:
            result['image_path'] = os.path.join('uploads', filename)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)