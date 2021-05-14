import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def motion(img):
    height, width = img.shape[:2]
    filtered_img = np.zeros(img.shape)

    _img = np.pad(img, [(2, 2), (2, 2), (0, 0)], "edge")
    k = np.eye(5) / 5
    for h in range(0, height):
        for w in range(0, width):
            for c in range(0, 3):
                filtered_img[h, w, c] = np.sum(_img[h:h+5, w:w+5, c] * k)

    return filtered_img.astype(np.uint8)


img = io.imread("./dataset/images/imori_256x256.png")
ans = motion(img)
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title("answer")
plt.imshow(ans)
plt.show()
