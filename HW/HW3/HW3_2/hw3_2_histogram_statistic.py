import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("hidden object.jpg", cv2.IMREAD_GRAYSCALE)  # shape: (665, 652, 3)
# img = cv2.imread("hidden object2.png", cv2.IMREAD_GRAYSCALE)  # shape: (665, 652, 3)

# img = cv2.resize(img, (652,665),interpolation=cv2.INTER_AREA)
cv2.imwrite("ori.png", img)


global_mean = np.mean(img)
print(global_mean)

global_std = np.std(img)
print(global_std)


kernel_size = 5
output_img = np.zeros_like(img)
h, w = img.shape
print(img.shape)
img = np.pad(img, ((kernel_size//2,kernel_size//2),(kernel_size//2,kernel_size//2)))
print(img.shape)

# k0, k1, k2, k3 = 0.16, 0.3, 0, 0.1
k0, k1, k2, k3 = 0.025,2, 0, 0.1


for y in np.arange(kernel_size//2, h - (kernel_size//2+1)):  #652
    for x in np.arange(kernel_size//2, w - kernel_size//2+1):
        
        area = img[(y-(kernel_size//2)) : (y+(kernel_size//2+1)), (x-(kernel_size//2)) : (x+(kernel_size//2+1))]
        local_mean = np.mean(area)
        local_std =np.std(area)

        if (local_mean >= k0 *global_mean) and (local_mean <= k1 *global_mean) and (local_std >= k2*global_std ) and (local_std <= k3*global_std ):
             
            output_img[y, x] = img[y, x]* 10
            # output_img[y, x] = img[y, x]* 5

        else:
            output_img[y, x] = img[y, x]


cv2.imwrite("ori_3_2_hist_stat.png", output_img)
# cv2.imwrite("3_2_hist_stat.png", output_img)


