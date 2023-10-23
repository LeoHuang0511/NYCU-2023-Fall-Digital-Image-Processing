import cv2
import matplotlib.pyplot as plt
import numpy as np

# img = cv2.imread("./checkerboard1024-shaded.tif",cv2.IMREAD_GRAYSCALE)
img = cv2.imread("./N1.bmp",cv2.IMREAD_GRAYSCALE)


print(img.shape)


# kernel_size = (257, 257)
# sigmaX, sigmaY = 64, 64
kernel_size = (57, 357)
sigmaX, sigmaY = 40, 60

shade = cv2.GaussianBlur(img,kernel_size, sigmaX=sigmaX, sigmaY=sigmaY)

# titles = ['original_1', 'estimated_shade_1', "divided_1"]
titles = ['original_2', 'estimated_shade_2', "divided_2"]


divided_img =  (img / shade)

divided_img = (divided_img-np.min(divided_img))*255/ (np.max(divided_img)-np.min(divided_img))

            
outputs = [img, shade, divided_img]

for idx, title in enumerate(titles):
    cv2.imwrite(f"./{title}.png", outputs[idx])
