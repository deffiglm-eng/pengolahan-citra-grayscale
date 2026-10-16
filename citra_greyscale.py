import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

# 1. Baca gambar berwarna (RGB)
# Ganti "gambar.jpg" dengan nama file gambarmu
img = iio.imread("deffi.jpg")

# Jika gambar punya channel alpha (RGBA), buang channel alpha
if img.ndim == 3 and img.shape[2] == 4:
    img = img[:, :, :3]

# Ubah ke grayscale (rumus luminance: 0.299 R + 0.587 G + 0.114 B)
gray = (0.299 * img[:, :, 0] +
        0.587 * img[:, :, 1] +
        0.114 * img[:, :, 2]).astype(np.uint8)

# 2. Hitung histogram: jumlah piksel untuk tiap intensitas 0-255
histogram = np.zeros(256, dtype=int)
for nilai in gray.flatten():
    histogram[nilai] += 1

# (opsional) verifikasi dengan numpy
hist_np, _ = np.histogram(gray, bins=256, range=(0, 256))
print("Hasil sama dengan np.histogram:", np.array_equal(histogram, hist_np))

# Jawaban (a): cetak jumlah piksel setiap intensitas
for i in range(256):
    print(f"Intensitas {i}: {histogram[i]} piksel")

print("Total piksel:", histogram.sum())

# Jawaban (b): cari intensitas dominan
dominan = np.argmax(histogram)
print(f"\nIntensitas dominan: {dominan} dengan {histogram[dominan]} piksel")

# 3. Plot histogram dengan Matplotlib
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.title("Gambar Grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.bar(range(256), histogram, width=1.0, color="gray")
plt.title("Histogram Grayscale")
plt.xlabel("Intensitas Piksel (0-255)")
plt.ylabel("Jumlah Piksel")
plt.xlim([0, 255])

plt.tight_layout()
plt.savefig("histogram.png", dpi=150)
plt.show()


