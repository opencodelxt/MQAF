import os
import cv2

path = r'D:\hqy\datasets\IQA\CLIVE\distorted_images'

for name in os.listdir(path):
    img = cv2.imread(os.path.join(path, name))
    H, W, _ = img.shape
    if H == 608 or W == 608:
    # print(H, ' ', W)
      print(name)