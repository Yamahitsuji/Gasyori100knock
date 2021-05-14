import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def median(img):
    height, width= img.shape[:2]
    filtered_img = np.zeros(img.shape)

    _img = np.pad(img, [(1, 1), (1, 1), (0, 0)], "edge")
    for h in range(0, height):
        for w in range(0, width):
            for c in range(0, 3):
                filtered_img[h, w, c] = np.median(_img[h:h+3, w:w+3, c])

    return filtered_img.astype(np.uint8)



img = io.imread("./dataset/images/imori_256x256_noise.png")
ans = median(img)
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title("answer")
plt.imshow(ans)
plt.show()
