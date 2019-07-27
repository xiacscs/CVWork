import cv2
import numpy as np

def ransacMatching(A,B):
    length = len(A)
    # 随机选择4对点作为内点，其它则为外点
    inliers = np.random.choice(range(length), size=4, replace=False)

    iterations = 0
    maxIteration = 1000
    threshold = 3
    while iterations < maxIteration:
        # 计算初始的转换矩阵
        inliersA = np.array([A[_] for _ in inliers])
        inliersB = np.array([B[_] for _ in inliers])
        H = cv2.findHomography(inliersA, inliersB)[0]

        new_inliers = []
        error = getError(A, B, H)
        for _ in range(length):
            if error[_] < threshold:
                new_inliers.append(_)

        if len(new_inliers) > len(inliers):
            inliers = new_inliers
        else:
            break
        iterations += 1

    return H

# 计算每对点的误差
def getError(A, B, H):
    length = len(A)
    err = np.zeros((length), dtype=A.dtype)
    for _ in range(length):
        a = np.array([*A[_], 1], dtype=A.dtype).reshape(3, 1)
        a_in_b = np.dot(H, a)
        dx = a_in_b[0, 0]/a_in_b[2, 0] - B[_, 0]
        dy = a_in_b[1, 0]/a_in_b[2, 0] - B[_, 1]
        err[_] = dx * dx + dy * dy

    return err


if __name__ == "__main__":
    # 读取图像
    left_img = cv2.imread('../img/left.jpg')
    right_img = cv2.imread('../img/right.jpg')

    # 转换成灰度图
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 通过SIFT算法获取两幅图像的特征点和描述符
    # sift = cv2.SIFT()
    hessian = 300
    sift = cv2.xfeatures2d.SIFT_create(hessian)  # 阈值越大能检测的特征就越少
    left_kp, left_des = sift.detectAndCompute(left_gray, None)
    right_kp, right_des = sift.detectAndCompute(right_gray, None)

    # FLANN算法获取特征匹配点
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 配置索引，密度树的数量为5
    search_params = dict(checks=50)  # 递归次数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(left_des, right_des, k=2)

    # 提取优秀的特征点
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 获取匹配的特征点的位置
    left_pts = np.array([left_kp[m.queryIdx].pt for m in good])
    right_pts = np.array([right_kp[m.trainIdx].pt for m in good])

    ransacMatching(left_pts, right_pts)
    # 生成变换矩阵, 左图到右图的单应矩阵,即左图中的点在右图中对应的位置
    # H, mask = cv2.findHomography(left_pts, right_pts, cv2.RANSAC, 5.0)
    H = ransacMatching(left_pts, right_pts)
    h1, w1 = left_gray.shape[0:2]
    h2, w2 = right_gray.shape[0:2]
    shift = np.array([[1.0, 0, w1], [0, 1.0, 0], [0, 0, 1.0]])
    # 获取左边图像到右边图像的投影映射关系，2幅图像放置一起，右边放置为右图，
    # 相当于右图平移左图的宽度，故此时单应矩阵应平移左图的宽度
    M = np.dot(shift, H)
    warpImg = cv2.warpPerspective(left_img, M, (w1 + w2, max(h1, h2)))
    cv2.imshow('left_img', warpImg)
    warpImg[0: h2, w1: w1 + w2] = right_img

    cv2.imshow('total_img', warpImg)
    cv2.imshow('left gray', left_img)
    cv2.imshow('right gray', right_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

