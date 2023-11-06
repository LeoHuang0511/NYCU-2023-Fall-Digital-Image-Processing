import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np



# matplotlib read image
img = cv2.imread("car-moire-pattern.tif",0)
print(img.shape)
# Output Images
plt.imsave('4_2_2_ori.png',img, cmap=cm.gray)


# Fourier Transform
fft = np.fft.fftshift(np.fft.fft2(img))
plt.imsave('4_2_2_fftshift.png',np.log(abs(fft) + 1), cmap='gray'), plt.title("fft_img")


mask = np.zeros_like(img)
# left column

mask = cv2.rectangle(mask, (44,41),(64,43), (255,255,255), -1)  
mask = cv2.rectangle(mask, (53,32),(55,52), (255,255,255), -1)  

mask = cv2.rectangle(mask, (44,83),(64,85), (255,255,255), -1)  
mask = cv2.rectangle(mask, (53,74),(55,94), (255,255,255), -1)  

mask = cv2.rectangle(mask, (46,164),(66,166), (255,255,255), -1)  
mask = cv2.rectangle(mask, (55,155),(57,175), (255,255,255), -1)  

mask = cv2.rectangle(mask, (46,204),(66,206), (255,255,255), -1)  
mask = cv2.rectangle(mask, (55,195),(57,215), (255,255,255), -1)  


# right column

mask = cv2.rectangle(mask, (100,39),(120,41), (255,255,255), -1)  
mask = cv2.rectangle(mask, (109,30),(111,50), (255,255,255), -1)  

mask = cv2.rectangle(mask, (100,79),(120,81), (255,255,255), -1)  
mask = cv2.rectangle(mask, (109,70),(111,90), (255,255,255), -1)  

mask = cv2.rectangle(mask, (103,160),(123,162), (255,255,255), -1)  
mask = cv2.rectangle(mask, (112,151),(114,171), (255,255,255), -1)  

mask = cv2.rectangle(mask, (103,201),(123,203), (255,255,255), -1)  
mask = cv2.rectangle(mask, (112,192),(114,212), (255,255,255), -1) 


mask = 255 - mask
mask = mask/np.max(mask)
plt.imsave('4_2_2_mask.png',mask, cmap='gray'), plt.title("mask")




new_fft = fft*mask
plt.imsave('4_2_2_modfft.png',np.log(abs(new_fft) + 1), cmap='gray'), plt.title("modify fft_img")



## inverse Fourier Transform
new_img = np.fft.ifft2(np.fft.ifftshift(new_fft))
plt.imsave('4_2_2_modimg.png',np.abs(new_img), cmap='gray'), plt.title("ifft_img")



