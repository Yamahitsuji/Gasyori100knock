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
    return np.clip(i, 0, 255).astype(np.uint8)


def hpf(g):
    h, w = g.shape
    h_half = h // 2
    w_half = w // 2
    switched_g = np.zeros_like(g)
    switched_g[:h_half, :w_half] = g[h_half:, w_half:]
    switched_g[:h_half, w_half:] = g[h_half:, :w_half]
    switched_g[h_half:, :w_half] = g[:h_half, w_half:]
    switched_g[h_half:, w_half:] = g[:h_half, :w_half]

    mask = np.ones((h, w, 3), np.uint8)
    cv2.circle(mask, (w_half, h_half), int(h_half*0.1), (0, 0, 0), -1)
    mask = mask[:, :, 0]
    switched_g *= mask

    out = np.zeros_like(g)
    out[:h_half, :w_half] = switched_g[h_half:, w_half:]
    out[:h_half, w_half:] = switched_g[h_half:, :w_half]
    out[h_half:, :w_half] = switched_g[:h_half, w_half:]
    out[h_half:, w_half:] = switched_g[:h_half, :w_half]
    return out


img = io.imread("./dataset/images/imori_128x128.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
g = dft(gray)
filtered = hpf(g)
ans = idft(filtered)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(gray, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("hpf")
plt.imshow(ans, cmap="gray")
plt.show()
