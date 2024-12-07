import imageio as img
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
image = img.imread(r'C:\Users\ASUS\Documents\Smester 5\Pengolahan citra digital\Praktikum Deteksi Tepi (Sobel)\pohon.jpg', mode='F')

# Sobel kernel untuk deteksi tepi
sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobelY = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# Menambahkan padding pada gambar
imgPad = np.pad(image, pad_width=1, mode='constant', constant_values=0)

# Matriks kosong untuk hasil gradient
Gx = np.zeros_like(image)
Gy = np.zeros_like(image)

for y in range(1, imgPad.shape[0] - 1):
    for x in range(1, imgPad.shape[1] - 1):
        area = imgPad[y-1:y+2, x-1:x+2]
        Gx[y-1, x-1] = np.sum(area * sobelX)
        Gy[y-1, x-1] = np.sum(area * sobelY)

# Menghitung magnitudo gradien
G = np.sqrt(Gx**2 + Gy*2)
G = (G / G.max()) * 255
G = np.clip(G, 0, 255)
G = G.astype(np.uint8)

# Menampilkan hasil
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(2, 2, 2)
plt.imshow(Gx, cmap='gray')
plt.title("Gradient X")

plt.subplot(2, 2, 3)
plt.imshow(Gy, cmap='gray')
plt.title("Gradient Y")

plt.subplot(2, 2, 4)
plt.imshow(G, cmap='gray')
plt.title("Gradient Magnitude")

plt.show()