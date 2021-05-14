import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def max_pooling(img, ksize_h=8, ksize_w=8):
    height, width = img.shape[:2]
    filtered_img = np.zeros(img.shape)
    for h in range(0, height, ksize_h):
        for w in range(0, width, ksize_w):
            for c in range(0, 3):
                filtered_img[h:h+ksize_h, w:w+ksize_w, c] = img[h:h+ksize_h, w:w+ksize_w, c].max()
    return filtered_img.astype(np.uint8)


img = io.imread("./dataset/images/imori_256x256.png")
ans = max_pooling(img)
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title("answer")
plt.imshow(ans)
plt.show()
