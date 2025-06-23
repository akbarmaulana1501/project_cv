import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import threading
import os

class DeepfakeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Deepfake Detector')
        self.root.configure(bg='#2C3E50')
        
        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton',
                        padding=10,
                        font=('Arial', 12),
                        background='#3498DB',
                        foreground='white')
        style.configure('TProgressbar',
                        troughcolor='#2C3E50',
                        background='#3498DB',
                        thickness=20)
        
        # Title
        title = tk.Label(root,
                        text='Deepfake Image Detector',
                        font=('Arial', 24, 'bold'),
                        bg='#2C3E50',
                        fg='white')
        title.pack(pady=20)
        
        # Image frame
        self.image_frame = tk.Frame(root,
                                   width=400,
                                   height=400,
                                   bg='#34495E',
                                   highlightbackground='#3498DB',
                                   highlightthickness=2)
        self.image_frame.pack(pady=20)
        self.image_frame.pack_propagate(False)
        
        # Image label
        self.image_label = tk.Label(self.image_frame,
                                   text='No image selected',
                                   bg='#34495E',
                                   fg='white')
        self.image_label.pack(expand=True)
        
        # Buttons frame
        btn_frame = tk.Frame(root, bg='#2C3E50')
        btn_frame.pack(pady=20)
        
        # Select button
        self.select_btn = ttk.Button(btn_frame,
                                    text='Select Image',
                                    command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=10)
        
        # Detect button
        self.detect_btn = ttk.Button(btn_frame,
                                    text='Detect',
                                    command=self.detect_deepfake,
                                    state='disabled')
        self.detect_btn.pack(side=tk.LEFT, padx=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(root,
                                       mode='indeterminate',
                                       length=300)
        
        # Result label
        self.result_label = tk.Label(root,
                                    text='',
                                    font=('Arial', 16, 'bold'),
                                    bg='#2C3E50',
                                    fg='white')
        self.result_label.pack(pady=20)
        
        self.image_path = None
        self.root.geometry('600x700')
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        
        if file_path:
            self.image_path = file_path
            # Load and resize image
            image = Image.open(file_path)
            image = image.resize((380, 380), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.detect_btn.configure(state='normal')
            self.result_label.configure(text='')
    
    def predict_image(self):
        try:
            # Load model
            model = tf.keras.models.load_model('deepfake_detector_model.h5')
            
            # Preprocess image
            img = tf.keras.utils.load_img(self.image_path, target_size=(128, 128))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            # Make prediction
            predictions = model.predict(img_array)
            score = predictions[0][0]
            
            # Calculate confidence
            confidence = 100 * (1 - score) if score < 0.5 else 100 * score
            result = 'REAL' if score < 0.5 else 'FAKE'
            
            # Update GUI in main thread
            self.root.after(0, self.update_result, result, confidence)
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
        finally:
            self.root.after(0, self.reset_ui)
    
    def detect_deepfake(self):
        self.detect_btn.configure(state='disabled')
        self.select_btn.configure(state='disabled')
        self.progress.pack(pady=10)
        self.progress.start(10)
        
        # Start prediction in separate thread
        thread = threading.Thread(target=self.predict_image)
        thread.daemon = True
        thread.start()
    
    def update_result(self, prediction, confidence):
        color = '#27AE60' if prediction == 'REAL' else '#E74C3C'
        self.result_label.configure(
            text=f'Prediction: {prediction}\nConfidence: {confidence:.2f}%',
            fg=color)
    
    def show_error(self, error_message):
        messagebox.showerror('Error', f'An error occurred: {error_message}')
    
    def reset_ui(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.detect_btn.configure(state='normal')
        self.select_btn.configure(state='normal')

def main():
    root = tk.Tk()
    app = DeepfakeDetectorGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()