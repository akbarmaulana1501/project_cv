import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Konfigurasi Awal & Persiapan Data ---

# Definisikan parameter-parameter utama
IMG_HEIGHT = 128  # Ubah sesuai kebutuhan, ukuran gambar yang lebih besar butuh lebih banyak memori
IMG_WIDTH = 128
BATCH_SIZE = 32  # Jumlah gambar yang diproses dalam satu iterasi
DATA_DIR = 'D:\cuy_universe\project_python\deepfake\Dataset' # GANTI DENGAN PATH DATASET ANDA!

# Memuat dataset training dari direktori
# Keras akan secara otomatis memberi label berdasarkan nama folder (real=0, fake=1)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    f'{DATA_DIR}/Train',
    labels='inferred',
    label_mode='binary',  # Karena hanya ada 2 kelas (real/fake)
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Memuat dataset validasi dari direktori
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    f'{DATA_DIR}/Validation',
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Mengambil nama kelas (misal: ['fake', 'real'])
class_names = train_dataset.class_names
print("Kelas yang ditemukan:", class_names)

# Optimasi performa data loading
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. Membangun Model CNN ---

# Kita akan membangun model CNN sederhana
model = Sequential([
    # Layer pertama: Rescaling untuk menormalisasi nilai piksel dari [0, 255] ke [0, 1]
    Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # Blok Konvolusi 1
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Blok Konvolusi 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Blok Konvolusi 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Meratakan output dari 2D menjadi 1D untuk dimasukkan ke layer Dense
    Flatten(),

    # Layer Dense (Fully Connected)
    Dense(128, activation='relu'),
    # Dropout untuk mencegah overfitting dengan mematikan beberapa neuron secara acak
    Dropout(0.5),

    # Layer Output
    # Menggunakan 1 neuron dengan aktivasi sigmoid untuk klasifikasi biner
    # Outputnya adalah probabilitas gambar tersebut adalah 'fake'
    Dense(1, activation='sigmoid')
])

# --- 3. Kompilasi Model ---

# Mengkonfigurasi model untuk training
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Menampilkan ringkasan arsitektur model
model.summary()

# --- 4. Melatih Model ---

EPOCHS = 10  # Jumlah berapa kali model akan melihat keseluruhan dataset
print("\n--- Memulai Pelatihan Model ---")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

print("--- Pelatihan Selesai ---")

# --- 5. Evaluasi dan Visualisasi Hasil ---

# Membuat plot untuk akurasi dan loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# --- 6. Menyimpan Model ---

model.save('deepfake_detector_model2.h5')
print("\nModel telah disimpan sebagai 'deepfake_detector_model2.h5'")


# --- 7. Fungsi untuk Prediksi Gambar Baru ---
def predict_image(image_path, saved_model):
    """Fungsi untuk memprediksi satu gambar."""
    # Memuat gambar
    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    # Mengubah gambar menjadi array numpy
    img_array = tf.keras.utils.img_to_array(img)
    # Menambahkan dimensi batch (karena model menerima input dalam batch)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Melakukan prediksi
    predictions = saved_model.predict(img_array)
    score = predictions[0][0]

    # Menentukan hasil prediksi
    confidence = 100 * (1 - score) if score < 0.5 else 100 * score
    
    # Asumsi: folder 'real' dilabeli 0, dan 'fake' dilabeli 1
    # Jika score < 0.5, lebih dekat ke 0 (real)
    if score < 0.5:
        print(f"Gambar ini diprediksi sebagai: REAL dengan keyakinan {confidence:.2f}%")
    else:
        print(f"Gambar ini diprediksi sebagai: FAKE dengan keyakinan {confidence:.2f}%")

# Contoh penggunaan fungsi prediksi
# Buat file baru untuk prediksi atau jalankan di bawah ini setelah training

# # Muat model yang sudah disimpan
# loaded_model = tf.keras.models.load_model('deepfake_detector_model.h5')

# # Ganti dengan path gambar yang ingin Anda tes
# test_image_path_real = '/path/to/your/dataset/validation/real/real_image_val_001.jpg'
# test_image_path_fake = '/path/to/your/dataset/validation/fake/fake_image_val_001.jpg'

# print("\n--- Tes Prediksi pada Gambar Asli ---")
# predict_image(test_image_path_real, loaded_model)

# print("\n--- Tes Prediksi pada Gambar Deepfake ---")
# predict_image(test_image_path_fake, loaded_model)