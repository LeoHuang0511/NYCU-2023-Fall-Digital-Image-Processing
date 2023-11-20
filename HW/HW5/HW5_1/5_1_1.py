import cv2
import matplotlib.pyplot as plt
import numpy as np


def ifft(input):
    ifft_img = np.fft.ifft2(np.fft.ifftshift(input))
    ifft_img = np.abs(ifft_img)
    ifft_img = (ifft_img/ifft_img.max()) *255

    
    return ifft_img




img = cv2.imread("Fig5.25.jpg")
plt.imsave("1_original.jpg",img, cmap='gray')


M, N, c  = img.shape
restored_inv_full = np.zeros_like(img)
restored_inv_rap = np.zeros_like(img)
restored_wie = np.zeros_like(img)


## degradation function H(u, v)
k = 0.0025
x, y = np.mgrid[0:M, 0:N]        
H = np.exp(-k * (((x-M/2)**2 + (y-N/2)**2)**(5/6)) )
plt.imsave("1_H.jpg",np.log(abs(H) + 1), cmap='gray')


## Butter filter
radio = 60
n = 10
x, y = np.mgrid[-M//2:M//2, -N//2:N//2]
butter = 1/(1+((x**2+y**2)/radio**2)**(n))   # 課本 p.282公式

fig = plt.figure()
plt.plot(butter.ravel())
plt.savefig("1_butter.jpg")

for i in range(c):
    G = np.fft.fftshift(np.fft.fft2(img[:,:,i]))
    plt.imsave("1_G.jpg",np.log(abs(G) + 1), cmap='gray')
    

    ## inverse filter full
    F_full = G/H
    
    ##inverse filter radially
    F = F_full*butter

    ## wiener filtering
    K = 0.0001
    weiner_term = (abs(H)**2/(abs(H)**2+K))
    F_wie = (G/H) * weiner_term

    restored_inv_full[:,:,i] = ifft(F_full)
    restored_inv_rap[:,:,i] = ifft(F)
    restored_wie[:,:,i] = ifft(F_wie)


plt.imsave("1_Inverse_Full.jpg",restored_inv_full, cmap='gray')
plt.imsave("1_Inverse_Radially.jpg",restored_inv_rap, cmap='gray')
plt.imsave("1_Wiener.jpg",restored_wie, cmap='gray')
