import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2


# 参考記事：https://qiita.com/haru1843/items/00de955790d3a22a217b
def getThreshold(img):
    max_variance = 0
    max_th = 0
    for th in range(1, 254):
        # thを閾値としたクラス１,2の画素配列
        c0 = img[img <= th]
        c1 = img[img > th]
        # クラス1,2の重み
        r0 = len(c0) / (len(c0) + len(c1))
        r1 = len(c1) / (len(c0) + len(c1))

        if len(c0) == 0 or len(c1) == 0:
            continue
        c0_avg = c0.mean()
        c1_avg = c1.mean()

        variance = r0 * r1 * ((c0_avg - c1_avg) ** 2)
        if variance > max_variance:
            max_variance = variance
            max_th = th

    return max_th


def gray2binary(gray_img, th):
    img = np.minimum(gray_img // th, 1) * 255
    return img.astype(np.uint8)


img = io.imread("./dataset/images/imori_256x256.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
th = getThreshold(gray_img)
binary_img = gray2binary(gray_img, th)

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
