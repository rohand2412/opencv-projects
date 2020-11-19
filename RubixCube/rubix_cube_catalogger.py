#!/usr/bin/env python3
"""This script recognizes hand gestures and number of the fingers held up"""

import cv2
from rubix_cube_utils import RubixCube

def main():
    """Main code"""
    frame = RubixCube.Frame("Frame")
    frame.init_color_data()

    try:
        while True:
            frame.capture_frame()
            frame.preprocessing()
            frame.find_contours()
            frame.processing()
            frame.imshow()

            RubixCube.check_for_quit_request()

    except RubixCube.Break:
        frame.get_camera().stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
