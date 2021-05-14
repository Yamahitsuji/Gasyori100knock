import matplotlib.pyplot as plt
import numpy as np
from skimage import io

# img = io.imread('./dataset/images/imori_256x256.png')
# gray_img = np.zeros((256, 256, 3))

# gray_img[:, :, 0] = 0.2126 * (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])
# gray_img[:, :, 1] = 0.7152 * (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])
# gray_img[:, :, 2] = 0.0722 * (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])
# gray_img = gray_img.astype(np.uint8)
#
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(img)
# ax[1].imshow(gray_img)
# plt.show()

def rgb2gray(img):
    _img = img.copy().astype(np.float32)
    gray = _img[..., 0] * 0.2126 + _img[..., 1] * 0.7152 + _img[..., 2] * 0.0722
    gray = np.clip(gray, 0, 255)
    return gray.astype(np.uint8)

img = io.imread('./dataset/images/imori_256x256.png')
img_gray = rgb2gray(img)
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title('input')
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title('answer')
plt.imshow(img_gray, cmap='gray')
plt.show()
