import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2


def sobel_filter_x(img):
    k = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return cv2.filter2D(img, -1, k)


def sobel_filter_y(img):
    k = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return cv2.filter2D(img, -1, k)


img = io.imread("./dataset/images/imori_256x256.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ans_x = sobel_filter_x(gray_img)
ans_y = sobel_filter_y(gray_img)
plt.figure(figsize=(12, 3))
plt.subplot(1, 4, 1)
plt.title("input")
plt.imshow(img)
plt.subplot(1, 4, 2)
plt.title("gray")
plt.imshow(gray_img, cmap="gray")
plt.subplot(1, 4, 3)
plt.title("answerX")
plt.imshow(ans_x, cmap="gray")
plt.subplot(1, 4, 4)
plt.title("answerY")
plt.imshow(ans_y, cmap="gray")
plt.show()
