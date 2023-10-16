import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./aerial_view.tif",cv2.IMREAD_GRAYSCALE)
print(img.shape)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])


fig = plt.figure()
plt.plot(hist)
plt.ylabel('# of pixel')
plt.xlabel('Bins')
plt.title("Grayscale Histogram (original)")
plt.savefig("3_1_original_hist.png")

cv2.imwrite("3_1_original.png",img)
