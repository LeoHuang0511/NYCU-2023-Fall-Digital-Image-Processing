import cv2
import matplotlib.pyplot as plt
import numpy as np

eq_img = cv2.imread("aerial_view.tif", cv2.IMREAD_GRAYSCALE)


hist_eq = cv2.calcHist([eq_img], [0], None, [256], [0, 256])

pdf = np.squeeze(hist_eq/eq_img.size)

c = 1/np.cumsum(np.arange(256)**0.4)[-1]
print("c:", c)  # c =  0.0005968222208221872
print("sum of pdf: ", np.sum(c*np.arange(256)**0.4))

G_source = np.round(np.cumsum(pdf)*255)

G_reference = np.round(np.cumsum(c*np.arange(256)**0.4)*255) # c =  0.0005968222208221872



match_img = np.empty_like(eq_img)

for i, sk in enumerate(G_source):
    for j, Gzq in enumerate(G_reference):
        if Gzq == sk:
            match_img[np.where(eq_img == i)] = j
            break
        elif (Gzq<sk) and (G_reference[j+1]>sk):
            value = np.array([j,j+1])
            dis = np.abs(sk - np.array([Gzq, G_reference[j+1]]))
            match_img[np.where(eq_img == i)] = value[np.argmin(dis)]
            break

hist_match = cv2.calcHist([match_img], [0], None, [256], [0, 256])
pdf_match = hist_match/match_img.size
cdf_match = np.uint8(np.cumsum(pdf_match)*255)

fig =  plt.figure()

plt.plot(hist_match)
plt.ylabel('# of pixel')
plt.xlabel('Bins')
plt.title("histogram matching", fontsize=15)
cv2.imwrite("3_1_hist_match.png",match_img)


plt.savefig("3_1_hist_match_hist.png")
cv2.imwrite("3_1_hist_match.png",match_img)


fig =  plt.figure()

line1, = plt.plot(cdf_match, label = "matched G(Zq)" )
line2, =plt.plot(G_source, label = "T(r)")
line3, =plt.plot(G_reference, label = "G(Zq)")

plt.legend(handles = [line1, line2, line3], loc='lower right')
plt.savefig("3_1_hist_match_cdf.png")





