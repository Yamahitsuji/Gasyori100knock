import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def affine(img, a):
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    a_inv = np.linalg.inv(a)
    for out_y in range(h):
        for out_x in range(w):
            x, y = np.dot(a_inv, np.array([out_x, out_y, 1])).astype(np.int)[:2]
            out[out_y, out_x] = img[y, x] if 0 <= y < h and 0 <= x < w else 0
    return out


img = io.imread("./dataset/images/imori_256x256.png")
rad = -np.pi / 6
ans1 = affine(img, np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]]))
h, w = img.shape[:2]
tx = w // 2 - int(np.cos(rad) * w // 2 - np.sin(rad) * h // 2)
ty = h // 2 - int(np.sin(rad) * w // 2 + np.cos(rad) * h // 2)
print(tx, ty)
ans2 = affine(img, np.array([[np.cos(rad), -np.sin(rad), tx], [np.sin(rad), np.cos(rad), ty], [0, 0, 1]]))

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.title("answer1")
plt.imshow(ans1)
plt.subplot(1, 3, 3)
plt.title("answer2")
plt.imshow(ans2)
plt.show()
