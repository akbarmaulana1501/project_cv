# 🧠 Deepfake Image Detection App

A Python-based application that detects manipulated (deepfake) content in images using a pre-trained TensorFlow model. The app features both a web interface (Flask) and a desktop GUI (Tkinter), with support for multithreading and real-time image analysis.

## 🚀 Features

- ✅ Deepfake detection using a CNN model
- 🖼️ Image preview and prediction result display
- 🌐 Web interface built with Flask
- 🪟 Desktop GUI with Tkinter
- 🔄 Multi-threaded image processing for better performance

---

## 👨‍💻 Author
Akbar Maulana

## 📁 Project Structure

deepfake/
├── Dataset/ # Dataset folder (ignored in Git)
├── app.py # Main Flask application
├── gui.py # Tkinter GUI application
├── model/ # Saved trained model files
├── static/ # Static assets for Flask (e.g., CSS, JS)
├── templates/ # HTML templates for Flask
├── venv/ # Python virtual environment (ignored)
├── requirements.txt # Dependency list
└── README.md # Project documentation


> Deepfake Detection App

## Penggunaan di local server

```bash
git clone https://github.com/akbarmaulana1501/project_cv.git


cd deepfake && pip install -r requirements.txt


python -m venv venv
venv\Scripts\activate 

python predict.py