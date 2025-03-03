import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread("0e1cced33fb99051009eee2c5cec4d5.png")

# 如果图像存在
if img is not None:
    # 转换颜色空间从BGR到RGB（OpenCV默认是BGR）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转换为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 边缘检测
    img_edges = cv2.Canny(img_blur, 100, 200)

    # 显示结果
    plt.figure(figsize=(12, 8))

    plt.subplot(221), plt.imshow(img_rgb)
    plt.title("原图"), plt.axis("off")

    plt.subplot(222), plt.imshow(img_gray, cmap="gray")
    plt.title("灰度图"), plt.axis("off")

    plt.subplot(223), plt.imshow(img_blur, cmap="gray")
    plt.title("高斯模糊"), plt.axis("off")

    plt.subplot(224), plt.imshow(img_edges, cmap="gray")
    plt.title("边缘检测"), plt.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("无法读取图像")
