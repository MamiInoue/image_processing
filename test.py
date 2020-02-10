# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('./bag/test.png')
img = cv2.imread('./data/y_pose7.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


img_array = np.array(img)

print(img_array.shape)

r_img = img_array[:,:,0]
g_img = img_array[:,:,1]
b_img = img_array[:,:,2]

print(r_img.shape)

cv2.imwrite("./test_data/r_img.jpg",r_img)
cv2.imwrite("./test_data/b_img.jpg",b_img)
cv2.imwrite("./test_data/g_img.jpg",g_img)

R_img = np.float32(r_img)
G_img = np.float32(g_img)
B_img = np.float32(b_img)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 148
ret,label,center=cv2.kmeans(R_img,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res_r = center[label.flatten()]
cv2.imwrite("./test_data/k_r.jpg",res_r)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 148
ret,label,center=cv2.kmeans(B_img,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res_b = center[label.flatten()]
cv2.imwrite("./test_data/k_b.jpg",res_b)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 148
ret,label,center=cv2.kmeans(G_img,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res_g = center[label.flatten()]
cv2.imwrite("./test_data/k_g.jpg",res_g)

ret2, img_otsu_r = cv2.threshold(res_r, 0, 255, cv2.THRESH_OTSU)
print("ret2: {}".format(ret2))
cv2.imwrite("./test_data/img_otsu_r.jpg",img_otsu_r)

ret2, img_otsu_b = cv2.threshold(res_b, 0, 255, cv2.THRESH_OTSU)
print("ret2: {}".format(ret2))
cv2.imwrite("./test_data/img_otsu_b.jpg",img_otsu_b)

ret2, img_otsu_g = cv2.threshold(res_g, 0, 255, cv2.THRESH_OTSU)
print("ret2: {}".format(ret2))
cv2.imwrite("./test_data/img_otsu_g.jpg",img_otsu_g)

kernel = np.ones((10,10),np.uint8)
op_r = cv2.morphologyEx(img_otsu_r, cv2.MORPH_OPEN, kernel)
op_g = cv2.morphologyEx(img_otsu_g, cv2.MORPH_OPEN, kernel)
op_b = cv2.morphologyEx(img_otsu_b, cv2.MORPH_OPEN, kernel)

bin_img = op_r + op_g + op_b

print(bin_img.shape)

op_bin = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
cl_bin = cv2.morphologyEx(op_bin, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("./test_data/res.jpg",cl_bin)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img_gray.shape)

test = cv2.addWeighted(src1=img_gray,alpha=0.6,src2=cl_bin,beta=0.4,gamma=0)
cv2.imwrite("./test_data/output.jpg",test)

titles = ['Original', 'Red','Blue', 'Green', 'Red_k', 'Blue_k',
          'Green_k', 'Red_ohtsu', 'Blue_ohtsu', 'Green_ohtsu',
          'Red_Opening', 'Green_Opening', 'Blue_Opening',
          'Bin', 'Closing', 'Output' ]

images = [img, r_img, b_img, g_img, res_r, res_b, res_g,
          img_otsu_r, img_otsu_b, img_otsu_g, op_r, op_g, op_b,
          bin_img, cl_bin, test]

for i in range(16):
    plt.subplot(4,4,i+1),plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.tight_layout()

plt.show()
