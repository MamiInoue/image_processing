#coding:utf-8
import cv2
from matplotlib import pyplot as plt
 
path = 'bag/red_9.png'          # 画像のパス
i = cv2.imread(path, 1)     # 画像読み込み
 
obj = i[2:110, 112:230]     # ROI抽出 [y1:y2, x1:x2]
i[157:265, 45:163] = obj    # ROI画像を元画像の異なる位置にペースト

# obj = i[65:225, 15:200]     # ROI抽出
# i[65:225, 185:370] = obj    # ROI画像を元画像の異なる位置にペースト

# matplotlibの表示に合わせてRGBの順番に色を並び替える。
i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
 
# ここからグラフ設定
# フォントの種類とサイズを設定する。
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'
fig = plt.figure()
ax1 = fig.add_subplot(111)
 
# 画像をプロット
ax1.imshow(i)
 
# 軸のラベルを設定する。
ax1.set_xlabel('x [pix]')
ax1.set_ylabel('y [pix]')
 
fig.tight_layout()
plt.show()
plt.close()