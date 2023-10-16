import cv2
import matplotlib.pyplot as plt

img = cv2.imread("aerial_view.tif", cv2.IMREAD_GRAYSCALE)


eq_img = cv2.equalizeHist(img)

hist = cv2.calcHist([eq_img], [0], None, [256], [0, 256])
equalized_hist = cv2.calcHist([eq_img], [0], None, [256], [0, 256])


cv2.imwrite("3_1_hist_eq.png", eq_img)

fig = plt.figure()
plt.plot(hist)
plt.ylabel('# of pixel')
plt.xlabel('Bins')
plt.title("histogram equalization", fontsize=15)

plt.savefig("3_1_hist_eq_hist.png")

