import matplotlib.pyplot as plt
import numpy as np
from skimage import io


def scaling_and_shift(img):
    return (50 / img.std() * (img - img.mean()) + 128).astype(np.uint8)


img = io.imread("./dataset/images/imori_256x256_dark.png")
ans = scaling_and_shift(img)

plt.figure(figsize=(12, 5))
plt.subplot(2, 2, 1)
plt.title("input")
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.title("answer")
plt.imshow(ans)
plt.subplot(2, 2, 3)
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.subplot(2, 2, 4)
plt.hist(ans.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()
