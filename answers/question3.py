import matplotlib.pyplot as plt
import numpy as np
from skimage import io


def rgb2gray(img):
    _img = img.copy().astype(np.float32)
    gray = _img[:, :, 0] * 0.2126 + _img[:, :, 1] * 0.7152 + _img[:, :, 2] * 0.0722
    gray = np.clip(gray, 0, 255)
    return gray.astype(np.uint8)


def gray2binary(gray_img):
    binary_img = np.zeros((gray_img.shape[0], gray_img.shape[1]))
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            binary_img[i, j] = 255 if gray_img[i, j] > 127 else 0
    # 模範解答はnumpyのブロードキャストを使っている。こちらの方が多分高速。
    # binary_img = np.minimum(_img // th, 1) * 255
    return binary_img.astype(np.uint8)


img = io.imread('./dataset/images/imori_256x256.png')
gray_img = rgb2gray(img)
binary_img = gray2binary(gray_img)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.title("gray")
plt.imshow(gray_img, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("binary")
plt.imshow(binary_img, cmap="gray")
plt.show()
