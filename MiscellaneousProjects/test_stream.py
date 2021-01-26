import cv2

cap = cv2.VideoCapture(0)

cap.set(3, 1920)
cap.set(4, 1080)

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.flip(frame, -1)
    
    #imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(imgGray, 127, 255, 0)
    #countours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(thresh, countours, -1, (0, 255, 0), 3)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()