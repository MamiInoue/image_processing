#coding:utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt

# 画像のパス 
input_file_path = "y_bag_k/y_pose3_k_16.png"  
output_file_path = "tmp.jpg"

# 画像のロード
img = cv2.imread(input_file_path)

# グレースケール変換
from_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 
# 特徴点抽出 (AKAZE)
akaze = cv2.AKAZE_create()
from_key_points, from_descriptions = akaze.detectAndCompute(from_img, None)
 
# キーポイントの表示
extraceted_img = cv2.drawKeypoints(from_img, from_key_points, None, flags=4)
def display(extraceted_img, output_file_path=output_file_path):
    cv2.imwrite(output_file_path, extraceted_img)
    plt.imshow(plt.imread(output_file_path))
    plt.axis('off')
    plt.show()
    
display(extraceted_img)

 