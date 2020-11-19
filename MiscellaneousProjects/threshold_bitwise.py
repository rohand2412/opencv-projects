import numpy as np
import cv2

sideLength = 512

img = np.zeros([sideLength, sideLength], np.uint8)
img = cv2.circle(img, (sideLength//2, sideLength//2), 50, (255), -1)

cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()