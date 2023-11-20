import cv2
import matplotlib.pyplot as plt
import numpy as np


def ifft(input):
    ifft_img = np.fft.ifft2(np.fft.ifftshift(input))
    ifft_img = np.abs(ifft_img)
    ifft_img = (ifft_img/ifft_img.max()) *255

    
    return ifft_img




img = cv2.imread("book-cover-blurred.tif")
plt.imsave("2_original.jpg",img, cmap='gray')

M, N, c  = img.shape
restored_inv_full = np.zeros_like(img)
restored_wie = np.zeros_like(img)


## degradation function H(u, v)
a = 0.1
b = 0.1
T = 1

u, v = np.mgrid[-M//2:M//2, -N//2:N//2]
H = (T/(np.pi*(u*a+v*b))*np.sin(np.pi*(u*a+v*b)))*np.exp(np.pi*(u*a+v*b)*-1j)
H[np.isnan(H)] = T
plt.imsave("2_H.jpg",np.log(abs(H) + 1), cmap='gray')




for i in range(c):
    G = np.fft.fftshift(np.fft.fft2(img[:,:,i]))
    plt.imsave("2_G.jpg",np.log(abs(G) + 1), cmap='gray')


    ## inverse filter full
    F_full = G/H
    
    ##inverse filter radially

    ## wiener filtering
    K = 0.0001
    weiner_term = (abs(H)**2/(abs(H)**2+K))
    F_wie = (G/H) * weiner_term

    restored_inv_full[:,:,i] = ifft(F_full)
    restored_wie[:,:,i] = ifft(F_wie)


plt.imsave("2_Inverse_Full.jpg",restored_inv_full, cmap='gray')
plt.imsave("2_Wiener.jpg",restored_wie, cmap='gray')

