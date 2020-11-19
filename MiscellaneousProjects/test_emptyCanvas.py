import numpy as np
import cv2

width = 500
height = 500

img = np.zeros([width, height, 3], np.uint8)

sideLength = 1
numOfCurrentSides = 0
color = (255, 0, 0)

xCurrent = int(width/2)
yCurrent = int(height/2)

for i in range(3000):
    xFuture = None
    yFuture = None
    
    if (xCurrent >= width) or (xCurrent <= 0) or (yCurrent >= height) or (yCurrent <= 0):
        break
    
    if (numOfCurrentSides % 4) == 0:
        xFuture = xCurrent + sideLength
        yFuture = yCurrent
        color = (255, 0, 0)
    elif (numOfCurrentSides % 4) == 1:
        xFuture = xCurrent
        yFuture = yCurrent - sideLength
        color = (0, 255, 0)
    elif (numOfCurrentSides % 4) == 2:
        xFuture = xCurrent - sideLength
        yFuture = yCurrent
        color = (0, 0, 255)
    elif (numOfCurrentSides % 4) == 3:
        xFuture = xCurrent
        yFuture = yCurrent + sideLength
        color = (255, 255, 0)

    lineThickness = 1
    img = cv2.line(img, (xCurrent, yCurrent), (xFuture, yFuture), color, lineThickness)
    
    xCurrent = xFuture
    yCurrent = yFuture
    sideLength += lineThickness*2
    numOfCurrentSides += 1

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()