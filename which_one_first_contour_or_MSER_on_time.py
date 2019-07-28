# comparision between using contours or MSER first on time they take,
# to find words or approximating to words for Document denoising
# for example: time by using contour then MSER: 4.744698407
#              time by using contour then MSER: 2.253134167

import cv2
import numpy as np
img = cv2.imread('image1.bmp') # source image

e1 = cv2.getTickCount() # contour then MSER
final = np.zeros([img.shape[0], img.shape[1],3], dtype=np.uint8)
final.fill(255)

imgCM = img.copy() # CM: contour (then) MSER
gray = cv2.cvtColor(imgCM , cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray , (5,5))
_, thresh = cv2.threshold(blur , 127 , 255 , cv2.THRESH_BINARY)
conyours , _ = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
max_w = 0.9 * img.shape[0]
max_h = 0.9 * img.shape[1]
min_w = 0.001 * img.shape[0]
min_h = 0.001 * img.shape[1]
for cnt in conyours:
    x , y , w , h = cv2.boundingRect(cnt)
    if w > max_w or w < min_w or h > max_h or h < min_h :
        continue
    final[y:y+h , x:x+w] = img[y:y+h , x:x+w]
    imgCM = cv2.rectangle(imgCM, (x, y), (x + w, y + h), (255, 255, 255), 10)

mser = cv2.MSER_create()
regions = mser.detectRegions(imgCM)
for p in regions[0]:  # convert them to list that can be used for polygons
        cv2.fillPoly(final, [cv2.convexHull(p.reshape(-1, 1, 2)) ], 0)
final = cv2.bitwise_or(final, img)


cv2.imwrite('which_one_first_contour_then_MSER.jpg' , final)
cv2.imwrite('which_one_first_contour_then.jpg' , imgCM)
e2 = cv2.getTickCount()
print('time by using contour then MSER: '+str((e2-e1)/cv2.getTickFrequency()))



e3 = cv2.getTickCount() # MSER then contour
final = np.zeros([img.shape[0], img.shape[1],3], dtype=np.uint8)
final.fill(255)
imgMC = img.copy() # MC: MSER (then) contour

mser = cv2.MSER_create()
gray = cv2.cvtColor(imgMC , cv2.COLOR_BGR2GRAY)
regions = mser.detectRegions(gray)
for p in regions[0]:  # convert them to list that can be used for polygons
        cv2.fillPoly(final, [cv2.convexHull(p.reshape(-1, 1, 2)) ], 0)
        cv2.fillPoly(imgMC, [cv2.convexHull(p.reshape(-1, 1, 2))], (255,255,255))
final = cv2.bitwise_or(final, img)

gray2 = cv2.cvtColor(imgMC , cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray2 , (5,5))
_, thresh = cv2.threshold(blur , 127 , 255 , cv2.THRESH_BINARY)
conyours , _ = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
max_w = 0.9 * img.shape[0]
max_h = 0.9 * img.shape[1]
min_w = 0.001 * img.shape[0]
min_h = 0.001 * img.shape[1]
for cnt in conyours:
    x , y , w , h = cv2.boundingRect(cnt)
    if w > max_w or w < min_w or h > max_h or h < min_h :
        continue
    final[y:y+h , x:x+w] = img[y:y+h , x:x+w]


cv2.imwrite('which_one_first_MSER_then_contour.jpg' , final)
e4 = cv2.getTickCount()
print('time by using contour then MSER: '+str((e4-e3)/cv2.getTickFrequency()))