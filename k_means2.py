#coding:utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('redonly.jpg')
output_file_path = "tmp.jpg"

Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 100
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
k_img = res.reshape((img.shape))

def display(k_img, output_file_path=output_file_path):
    cv2.imwrite(output_file_path, k_img)
    plt.imshow(plt.imread(output_file_path))
    plt.axis('off')
    plt.show()
    
display(k_img)

# plt.imshow(cv2.cvtColor(res3, cv2.COLOR_BGR2RGB))

# まずはgray scaleへ変換する
# gray = cv2.cvtColor(res3,cv2.COLOR_BGR2GRAY)
# plt.imshow(gray)
# 
# plt.imshow(gray,cmap='gray')

# Otsu法で画像を二値化する
# thresh,bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# plt.imshow(bin_img,cmap='gray')
# print ('大津法の二値化によって決定した閾値:',thresh)
# 
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(bin_img,cv2.MORPH_OPEN,kernel,iterations = 2)
# plt.imshow(opening,cmap='gray')

# モルフォロジー演算のDilationを使う
# sure_bg = cv2.dilate(opening,kernel,iterations=2)
# plt.imshow(sure_bg,cmap='gray')
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# plt.imshow(dist_transform)
# 
# ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
# print ('閾値（距離変換で得られた値の最大値×0.5）:',ret)
# plt.imshow(sure_fg,cmap='gray')
# 
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
# plt.imshow(unknown,cmap='gray')

# foregroundの1オブジェクトごとにラベル（番号）を振っていく
# ret, markers = cv2.connectedComponents(sure_fg)
# plt.imshow(markers)

#  markersのデータの中をのぞいみている
# np.unique(markers,return_counts=True)

# markers = markers+1
# np.unique(markers,return_counts=True)
# 
# markers[unknown==255] = 0
# plt.imshow(markers)
# 
# np.unique(markers,return_counts=True)

# "ヒント"であるmarkersをwatershedに適応する
# markers = cv2.watershed(img,markers)
# plt.imshow(markers)

# 境界の領域を赤で塗る
# img[markers == -1] = [255,0,0]
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

