import cv2
import numpy as np


# 中值滤波
def medianBlur(img, kernel, padding_way):
    # img & kernel is List of List; padding_way a string
    h = img.shape[0]
    w = img.shape[1]
    kernel_size = kernel.shape

    # 扩展图像
    padding_size_h = kernel_size[0] // 2
    padding_size_w = kernel_size[1] // 2
    extend_img_h = padding_size_h * 2 + h
    extend_img_w = padding_size_w * 2 + w
    extend_img = img
    padding_value = []
    # 填充行
    for i in [h-1, 0]:
        if padding_way == 'ZERO':
            padding_value = 0
        elif padding_way == 'REPLICA':
            padding_value = img[i, :]
        for j in range(padding_size_h):
            if i == h - 1:
                i = h
            extend_img = np.insert(extend_img, i, padding_value, axis=0)
            # print(i, extend_img)
    # 填充列
    for i in [w-1, 0]:
        if padding_way == 'ZERO':
            padding_value = 0
        elif padding_way == 'REPLICA':
            padding_value = extend_img[:, i]
        for j in range(padding_size_w):
            if i == w - 1:
                i = w
            extend_img = np.insert(extend_img, i, padding_value, axis=1)
            # print(extend_img)

    # 循环遍历
    for i in range(padding_size_h, extend_img_h - padding_size_h):
        for j in range(padding_size_w, extend_img_w - padding_size_w):
            # print(i, j)
            windows = []
            for m in range(kernel_size[0]):
                for n in range(kernel_size[1]):
                    windows = extend_img[i - padding_size_h: i + padding_size_h + 1,
                                         j - padding_size_w: j + padding_size_w + 1]

            # 寻找中值
            # print(windows)
            windows = windows.ravel()
            windows.sort()
            # 修改图像中对应位置的值
            img[i-padding_size_h, j-padding_size_w] = windows[kernel.size // 2]
    return img


# 灰度图像
img_gray = cv2.imread('../img/lena.jpg', 0)
kernel_smooth = np.zeros((3, 3))
img_medianBlur = medianBlur(img_gray.copy(), kernel_smooth, 'ZERO')

cv2.imshow('lena', img_gray)
cv2.imshow('median blur lena', img_medianBlur)
cv2.waitKey(0)
cv2.destroyAllWindows()

