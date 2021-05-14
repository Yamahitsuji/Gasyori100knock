import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2

img = io.imread("./dataset/images/imori_256x256.png")
ans = img // 64 * 64 + 32

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title("answer")
plt.imshow(ans)
plt.show()
