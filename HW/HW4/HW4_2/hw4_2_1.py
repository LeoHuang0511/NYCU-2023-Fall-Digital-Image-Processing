import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_max_idx(array, num):
    a = array.copy()
    c = np.zeros(num)

    for i in range(num):
        c[i] = np.max(a)
        if i > 0:
            c[i] = np.max(np.max(a[a<c[i-1]]))
        if i == 0:
            b = np.stack(np.where(a==c[i]),axis=1)
        else:
            b = np.vstack((b,np.stack(np.where(a==c[i]),axis=1)))
    return b

img = cv2.imread("./astronaut-interference.tif", 0)
w, h = img.shape
print(img.shape)
plt.imsave('4_2_1_ori.png', img, cmap='gray')

fft = np.fft.fft2(img)
fft = np.fft.fftshift(fft)
res = np.log(np.abs(fft))
plt.imsave('4_2_1_fftshift.png', res, cmap='gray')
max_idx = find_max_idx(res, 3)
index = max_idx[1:3]
print(index)

mask = np.ones_like(img) / 1.
mask[index[:,0],index[:,1]] = 0.
print(np.where(mask==0))
plt.imsave('4_2_1_mask.png',mask, cmap='gray')


new_fft = fft * mask
plt.imsave('4_2_1_modfft.png', np.log(np.abs(fft))*mask, cmap='gray')
plt.show()

ifft = np.fft.ifftshift(new_fft)
ifft = np.fft.ifft2(ifft)
plt.imsave( '4_2_1_modimg.png',np.abs(ifft), cmap='gray')
