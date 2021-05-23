import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2


def dft(img):
    h, w = img.shape[:2]
    g = np.zeros((h, w), dtype=complex)
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    for k in range(w):
        for l in range(h):
            g[l, k] = np.sum(img * np.exp(-2j * np.pi * (k*x/w + l*y/h)))
    return g


def idft(g):
    h, w = g.shape
    i = np.zeros((h, w), dtype=np.float)
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    for k in range(w):
        for l in range(h):
            i[l, k] = np.abs(np.sum(g * np.exp(2j * np.pi * (x*k/w + y*l/h)))) / (h * w)
    return i.astype(np.uint8)


img = io.imread("./dataset/images/imori_128x128.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
g = dft(gray)
out = idft(g)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(gray, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("dft -> idft")
plt.imshow(out, cmap="gray")
plt.show()
