import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2


def emboss_filter(img):
    k = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    return cv2.filter2D(img, -1, k)


img = io.imread("./dataset/images/imori_256x256.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ans = emboss_filter(gray_img)
plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.title("gray")
plt.imshow(gray_img, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("answer")
plt.imshow(ans, cmap="gray")
plt.show()
