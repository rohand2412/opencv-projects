#!/usr/bin/env python3
"""This script identifies and localizes coins in a video feed"""

import math
import cv2
import numpy as np

def nothing(_):
    """Empty function that does nothing"""

def main():
    """Main code"""
    cv2.namedWindow("Frame Circles")
    cap = cv2.VideoCapture(0)

    cv2.createTrackbar("Penny B Thresh", "Frame Circles", 0, 255, nothing)
    cv2.createTrackbar("Penny G Thresh", "Frame Circles", 40, 255, nothing)
    cv2.createTrackbar("Penny R Thresh", "Frame Circles", 58, 255, nothing)

    while True:
        _, frame = cap.read()

        kernel_size = 50
        min_dist = 100
        min_radius = 40
        max_radius = 500
        penny_b_thresh = cv2.getTrackbarPos("Penny B Thresh", "Frame Circles")
        penny_g_thresh = cv2.getTrackbarPos("Penny G Thresh", "Frame Circles")
        penny_r_thresh = cv2.getTrackbarPos("Penny R Thresh", "Frame Circles")

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(frame_gray, (kernel_size*2+1, kernel_size*2+1),
                                      cv2.BORDER_DEFAULT)
        frame_circs = cv2.HoughCircles(frame_blur, cv2.HOUGH_GRADIENT, 0.9, min_dist,
                                       param1=50, param2=30, minRadius=min_radius,
                                       maxRadius=max_radius)

        if frame_circs is not None:
            frame_circs_rounded = np.uint16(np.around(frame_circs))
            frame_circs_rounded = frame_circs_rounded[:,
                                                      frame_circs_rounded[0, :, 2].argsort()[::-1],
                                                      :]

            frame_circs_drawn = frame.copy()
            penny_counter = 0
            dime_counter = 0
            for i, circ in enumerate(frame_circs_rounded[0, :]):
                if i < 4:
                    cv2.circle(frame_circs_drawn, (circ[0], circ[1]), circ[2], (0, 255, 0), 3)
                else:
                    if penny_counter < 2:
                        frame_penny = frame[circ[1]-circ[2]:circ[1]+circ[2],
                                            circ[0]-circ[2]:circ[0]+circ[2]].copy()
                        frame_penny_side = frame_penny.shape[0]
                        new_side_length = 10
                        frame_penny_scale = new_side_length/frame_penny_side
                        frame_penny = cv2.resize(frame_penny, (new_side_length, new_side_length))
                        for j in range(frame_penny.shape[0]):
                            for k in range(frame_penny.shape[1]):
                                if math.hypot(k-circ[2]*frame_penny_scale, j-circ[2]*frame_penny_scale) > circ[2]*frame_penny_scale:
                                    frame_penny[j, k] = (0, 0, 0)

                        penny_area = (math.pi * circ[2]**2) * frame_penny_scale**2
                        frame_penny_b_total = np.sum(frame_penny[:, :, 0])
                        frame_penny_g_total = np.sum(frame_penny[:, :, 1])
                        frame_penny_r_total = np.sum(frame_penny[:, :, 2])
                        frame_penny_b_avg = frame_penny_b_total/penny_area
                        frame_penny_g_avg = frame_penny_g_total/penny_area
                        frame_penny_r_avg = frame_penny_r_total/penny_area

                        if frame_penny_b_avg >= penny_b_thresh and frame_penny_g_avg >= penny_g_thresh and frame_penny_r_avg >= penny_r_thresh:
                            cv2.circle(frame_circs_drawn, (circ[0], circ[1]), circ[2], (255, 0, 0),
                                       3)
                            penny_counter += 1
                        else:
                            if dime_counter < 2:
                                cv2.circle(frame_circs_drawn, (circ[0], circ[1]), circ[2],
                                           (0, 0, 255), 3)
                                dime_counter += 1
                            else:
                                cv2.circle(frame_circs_drawn, (circ[0], circ[1]), circ[2],
                                           (255, 0, 0), 3)
                                penny_counter += 1
                    else:
                        cv2.circle(frame_circs_drawn, (circ[0], circ[1]), circ[2], (0, 0, 255), 3)

            cv2.imshow("Frame Circles", frame_circs_drawn)

        cv2.imshow("Frame Blur", frame_blur)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
