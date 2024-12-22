import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_lpf(image_path, radius):
    # a. Membaca input, memisahkan channel RGB, dan mengubah ke domain frekuensi
    image = cv2.imread("saul.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konversi ke RGB

    channels = cv2.split(image)  # Memisahkan channel RGB

    filtered_channels = []

    for channel in channels:
        # Transformasi Fourier
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        # b. Membuat filter LPF
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, -1)  # Membuat lingkaran mask

        # c. Mengaplikasikan filter
        fshift_filtered = fshift * mask

        # Transformasi invers Fourier
        f_ishift = np.fft.ifftshift(fshift_filtered)
        filtered_channel = np.fft.ifft2(f_ishift)
        filtered_channel = np.abs(filtered_channel)

        filtered_channels.append(filtered_channel)

    # d. Mengembalikan hasil filter ke domain spatial
    result_image = cv2.merge(filtered_channels)

    # Normalisasi untuk memastikan nilai berada dalam rentang [0, 255]
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)

    # Menampilkan hasil
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Filtered Image")
    plt.imshow(result_image)
    plt.axis('off')

    plt.show()

# Contoh penggunaan
apply_lpf("face.jpg", radius=20)
