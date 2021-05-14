import numpy as np
import matplotlib.pyplot as plt
from skimage import io


img = io.imread("./dataset/images/imori_256x256_dark.png")

# https://pythondatascience.plavox.info/matplotlib/%E3%83%92%E3%82%B9%E3%83%88%E3%82%B0%E3%83%A9%E3%83%A0
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()
