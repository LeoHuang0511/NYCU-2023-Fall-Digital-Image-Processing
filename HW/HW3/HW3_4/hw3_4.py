import cv2
import numpy as np


# img = cv2.imread("./Bodybone.bmp")
img = cv2.imread("./fish.jpg")


laplacian_kernel = np.array([[-1,-1,-1],
                            [-1, 8,-1],
                            [-1,-1,-1]])
                            
laplacian = cv2.filter2D(img, cv2.CV_16S, laplacian_kernel)
laplacian_save = (255*((laplacian-np.min(laplacian))/ (np.max(laplacian)-np.min(laplacian)))).astype(np.uint8) 

lap_sharpend = img + laplacian
sobel_x = cv2.Sobel(img , cv2.CV_16S, 1, 0, ksize=3) 
sobel_y = cv2.Sobel(img , cv2.CV_16S, 0, 1, ksize=3) 
sobel = np.abs(sobel_x) + np.abs(sobel_y)

# box_kernel = np.ones((5,5),np.float32)/25
box_kernel = np.ones((3,3))*(1/9)
sobel_blur = cv2.filter2D(sobel, cv2.CV_16S, box_kernel)

mask = (laplacian * sobel_blur)
mask[mask<0] = 0
mask = (255*((mask - np.min(mask)) / (np.max(mask) - np.min(mask)))).astype(np.uint8)
soblap_sharpened = img + mask


# c = 1
# gamma = 0.5
c = 2
gamma = 0.6
powerlaw = soblap_sharpened**gamma*c
powerlaw = (255*((powerlaw - np.min(powerlaw)) / (np.max(powerlaw) - np.min(powerlaw)))).astype(np.uint8)


# titles = ['original_1', 'lap_1', 'lap_sharp1', 'sobel1', 'sobel_blur1', 'mask1', 'soblap_sharpened1', 'power_law1']
titles = ['original_2', 'lap_2', 'lap_sharp2', 'sobel2', 'sobel_blur2', 'mask2', 'soblap_sharpened2', 'power_law2']

outputs = [img, laplacian_save, lap_sharpend, sobel, sobel_blur, mask, soblap_sharpened, powerlaw]

for idx, title in enumerate(titles):
    cv2.imwrite(f"./{title}.png", outputs[idx])