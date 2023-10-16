import cv2
import numpy as np
import matplotlib.pyplot as plt



img = cv2.imread("hidden object.jpg", cv2.IMREAD_GRAYSCALE)  # shape: (665, 652, 3)
# img = cv2.imread("hidden object2.png", cv2.IMREAD_GRAYSCALE)  # shape: (665, 652, 3)

# img = cv2.resize(img, (652,665),interpolation=cv2.INTER_AREA)




kernel_size = 13
# kernel_size = 5
stride = 1
output_img = np.empty_like(img)
h, w = img.shape
print(img.shape)
img = np.pad(img, ((kernel_size//2,kernel_size//2),(kernel_size//2,kernel_size//2)))
print(img.shape)

for y in np.arange(kernel_size//2, h - (kernel_size//2)+1, stride):  
    for x in np.arange(kernel_size//2, w - kernel_size//2 +1, stride):
        y1 = (y-(kernel_size//2))
        y2 = (y+(kernel_size//2)+1)
        x1 = (x-(kernel_size//2))
        x2 = (x+(kernel_size//2)+1)
        area = img[y1 : y2, x1 : x2 ]
        equalized_area = cv2.equalizeHist(area)
        output_img[y, x] = equalized_area[kernel_size//2, kernel_size//2]  

       
print(output_img.shape)
cv2.imwrite("ori_3_2_local_enhance.png", output_img)
# cv2.imwrite("3_2_local_enhance.png", output_img)



