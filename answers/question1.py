import cv2
import matplotlib.pyplot as plt
from skimage import io

rgb_img = io.imread('./dataset/images/imori_256x256.png')
bgr_img = rgb_img[:, :, ::-1]

plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
plt.title('input')
plt.imshow(rgb_img)
plt.subplot(1, 2, 2)
plt.title('answer')
plt.imshow(bgr_img)
plt.show()
