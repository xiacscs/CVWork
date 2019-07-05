import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

# 显示灰度图像
img_gray = cv2.imread('./img/lena.jpg', 0)
cv2.imshow('lena', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(img_gray)
#print(img_gray.dtype) # 数据类型
#print(img_gray.shape)  # h, w

# 显示彩色图像
img = cv2.imread('./img/lena.jpg')
cv2.imshow('lena', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image crop 图像裁剪
img_crop = img[0:300, 0:300, :]
cv2.imshow('lena_crop', img_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 获取B、G、R三个通道的数据
B, G, R = cv2.split(img)
#print(B)
cv2.imshow('lena_b', B)
cv2.imshow('lena_g', G)
cv2.imshow('lena_r', R)
cv2.waitKey(0)
cv2.destroyAllWindows()

#######################################
# change color 改变图像颜色
def random_light_color(img):
    # 获取每个通道的值
    B, G, R = cv2.split(img)

    # 随机整数
    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:  # 限制在0-255之间
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    # 重新合成彩色图像
    img_merge = cv2.merge((B, G, R))
    return img_merge


img_random_color = random_light_color(img)
cv2.imshow('img_random_color', img_random_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
#######################################

# gamma correction 伽马校正  将暗图像变亮
img_dark = cv2.imread('C:/Users/xm/Desktop/dark.jpg')


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)  # 归一化 指数运算 反归一化
    table = np.array(table).astype('uint8')
    return cv2.LUT(image, table)


img_brighter = adjust_gamma(img_dark, 2)
cv2.imshow('img_dark', img_dark)
cv2.imshow('img_brighter', img_brighter)
cv2.waitKey(0)
cv2.destroyAllWindows()
#######################################

# histogram 直方图
# 调整图像大小
img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[0]*0.5), int(img_brighter.shape[1]*0.5)))
plt.hist(img_brighter.flatten(), 256, [0, 256], color='r')
img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)  # BGR-YUV
# equalize the histogram of the Y channel Y通道
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])   # 对Y通道的数据进行直方图均衡化
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)   # y: luminance(明亮度), u&v: 色度饱和度
cv2.imshow('Color input image', img_small_brighter)
cv2.imshow('Histogram equalized', img_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
#######################################

# scale+rotation+translation = similarity transform  相似变换
M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 0.5) # center, angle, scale  获得转换矩阵
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lena', img_rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()
#######################################

# Affine Transform 仿射变换
rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])  # 原始图像中三个点的位置
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])  # 相对应的输出图像中三个点的位置

M = cv2.getAffineTransform(pts1, pts2)  # 获得变换矩阵2*3
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('affine lena', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
#######################################

# perspective transform 透视变换


def random_warp(img):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]) # 原始图像中四个点的位置
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]]) # 相对应的输出图像中四个点的位置
    M_warp = cv2.getPerspectiveTransform(pts1, pts2) # 获得变换矩阵3*3
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp


M_warp, img_warp = random_warp(img)
cv2.imshow('perspective lena', img_warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
