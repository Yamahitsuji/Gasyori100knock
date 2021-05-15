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
ans1 = affine(img, np.array([[1.3, 0, 0], [0, 0.8, 0], [0, 0, 1]]))
ans2 = affine(img, np.array([[1.3, 0, 30], [0, 0.8, -30], [0, 0, 1]]))

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
