import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# 問題の導出の逆変換は並行移動後に変換している。一方、変換後に平行移動の場合は(x, y) = A^-1{(x, y) - (tx, ty)}となる。
# 問題文の順変換は変換後に移動しているのに対し、逆変換は移動後に変換した場合の式となっているので矛盾している。
# ただしこの問題は平行移動のみであり、Aは単位行列なのでどちらも同じになる。


def affine(img, a):
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    a_inv = np.linalg.inv(a)
    for out_y in range(h):
        for out_x in range(w):
            x, y = np.dot(a_inv, np.array([out_x, out_y, 1])).astype(np.int)[:2]
            out[out_y, out_x] = img[y, x] if  0 <= y < h and 0 <= x < w else 0
    return out


img = io.imread("./dataset/images/imori_256x256.png")
ans2 = affine(img, np.array([[1, 0, 30], [0, 1, -30], [0, 0, 1]]))

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title("answer")
plt.imshow(ans)
plt.show()
