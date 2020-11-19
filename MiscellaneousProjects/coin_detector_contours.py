#!/usr/bin/env python3
"""This script identifies and localizes coins in a video feed"""

import cv2
import numpy as np

def nothing(_):
    """Empty function that does nothing"""

def main():
    """Main code"""
    cv2.namedWindow("Frame Canny")
    cap = cv2.VideoCapture(0)

    cv2.createTrackbar("Kernel size", "Frame Canny", 1, 50, nothing)
    cv2.createTrackbar("Erode Iterations", "Frame Canny", 1, 10, nothing)
    cv2.createTrackbar("Dilate Iterations", "Frame Canny", 1, 10, nothing)
    while True:
        _, frame = cap.read()

        kernel_size = cv2.getTrackbarPos("Kernel size", "Frame Canny")
        erode_iterations = cv2.getTrackbarPos("Erode Iterations", "Frame Canny")
        dilate_iterations = cv2.getTrackbarPos("Dilate Iterations", "Frame Canny")
        morph_kernel = np.ones((2, 2), np.uint8)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(frame_gray, (kernel_size*2+1, kernel_size*2+1), 
                                      cv2.BORDER_DEFAULT)
        frame_threshold = cv2.adaptiveThreshold(frame_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
        frame_erode = cv2.erode(frame_threshold, morph_kernel, iterations=erode_iterations)
        frame_dilate = cv2.dilate(frame_erode, morph_kernel, iterations=dilate_iterations)
        frame_canny = cv2.Canny(frame_dilate.copy(), 100, 200)
        contours, _ = cv2.findContours(frame_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        cv2.imshow("Frame Canny", frame_canny)
        cv2.imshow("Frame Threshold", frame_threshold)
        cv2.imshow("Frame Dilate", frame_dilate)
        cv2.imshow("Frame Blur", frame_blur)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
