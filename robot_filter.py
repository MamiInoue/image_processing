#!/usr/bin/env python
# coding: utf-8

import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

class Cable_filter(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.input_sub = rospy.Subscriber('/photoneo_center/rgb_texture', Image, self.callback)
        self.cable_filter = rospy.Publisher('/photoneo_center/cable_filter_image',Image,queue_size=10)
        self.img = Image()

    def callback(self,image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.img = image
        cl_bin = Image()

        print('Subscribed photoneo_image')

        # ROI抽出 [y1:y2, x1:x2]
        roi_img = cv_image[2:615, 712:1330]
        # roi_img = cv_image[2:640, 712:1330]

        print('Successed roi')

        gray_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gamma = 1.5

        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        # Look up tableを使って画像の輝度値を変更
        gamma_img = cv2.LUT(gray_img, lookUpTable)
        
        print('Successed gamma')

        # hist_img = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        blur_img = cv2.GaussianBlur(gray_img,(3,3),0)
        print('Successed gaussian_filter')
        ret, ohtsu_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_OTSU)
        print('Processed binarization')
        kernel = np.ones((10,10),np.uint8)
        op_bin = cv2.morphologyEx(ohtsu_img, cv2.MORPH_OPEN, kernel)
        cl_bin = cv2.morphologyEx(op_bin, cv2.MORPH_CLOSE, kernel)

        self.cable_filter.publish(self.bridge.cv2_to_imgmsg(cl_bin, "mono8"))

        print('Published result_image')

if __name__ == '__main__':
    rospy.init_node('cable_filter_server')

    pp = Cable_filter()
    rospy.spin()