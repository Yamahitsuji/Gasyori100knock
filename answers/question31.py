import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def affine(img, a):
    h, w = img.shape[:2]
    out = np.zeros((300, 300, 3), dtype=int)
    out_h, out_w = out.shape[:2]
    a_inv = np.linalg.inv(a)
    for out_y in range(out_h):
        for out_x in range(out_w):
            x, y = np.dot(a_inv, np.array([out_x, out_y, 1])).astype(np.int)[:2]
            out[out_y, out_x] = img[y, x] if 0 <= y < h and 0 <= x < w else 0
    return out


img = io.imread("./dataset/images/imori_256x256.png")
h, w = img.shape[:2]
x_sharing = affine(img, np.array([[1, 30/h, 0], [0, 1, 0], [0, 0, 1]]))
y_sharing = affine(img, np.array([[1, 0, 0], [30/w, 1, 0], [0, 0, 1]]))
xy_sharing = affine(img, np.array([[1, 30/h, 0], [30/w, 1, 0], [0, 0, 1]]))

plt.figure(figsize=(12, 3))
plt.subplot(1, 4, 1)
plt.title("input")
plt.imshow(img)
plt.subplot(1, 4, 2)
plt.title("X-sharing(dx = 30)")
plt.imshow(x_sharing)
plt.subplot(1, 4, 3)
plt.title("Y-sharing(dy = 30)")
plt.imshow(y_sharing)
plt.subplot(1, 4, 4)
plt.title("dx = 30, dy = 30")
plt.imshow(xy_sharing)
plt.show()
