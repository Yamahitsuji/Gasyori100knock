import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2


def equalize(img):
    x_max = img.max()
    s = img.size
    h = np.zeros(256)
    for i in range(256):
        h[i] = np.count_nonzero(img == i)

    out = np.zeros_like(img)
    for i in range(256):
        out[img == i] = x_max / s * h[:i+1].sum()
    return out.astype(np.uint8)


img = io.imread("./dataset/images/imori_256x256_dark.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ans = equalize(gray)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title("gray")
plt.imshow(gray, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("answer")
plt.imshow(ans, cmap="gray")
plt.show()
