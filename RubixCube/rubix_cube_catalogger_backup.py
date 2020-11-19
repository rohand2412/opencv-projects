#!/usr/bin/env python3
"""This script recognizes hand gestures and number of the fingers held up"""

import cv2
import numpy as np
import my_python_modules as modules

def main():
    """Main code"""
    window_detection_name = "Frame"
    cv2.namedWindow(window_detection_name)
    cap = cv2.VideoCapture(0)
    colors_max_values = [360//2, 255, 255]
    colors_names = ["H", "S", "V"]
    colors_keys = ["orange", "green", "yellow", "white", "blue", "red"]
    colors = {"orange": modules.ColorTracker(colors_max_values, colors_names,
                                             window_detection_name,
                                             [(7, 12), (225, 255), (100, 255)]),
              "green": modules.ColorTracker(colors_max_values, colors_names,
                                            window_detection_name,
                                            [(45, 65), (150, 255), (20, 100)]),
              "yellow": modules.ColorTracker(colors_max_values, colors_names,
                                             window_detection_name,
                                             [(17, 26), (120, 255), (75, 255)]),
              "white": modules.ColorTracker(colors_max_values, colors_names,
                                            window_detection_name,
                                            [(0, 60), (0, 90), (100, 190)]),
              "blue": modules.ColorTracker(colors_max_values, colors_names,
                                           window_detection_name,
                                           [(110, 140), (0, 255), (50, 100)])}

    initial_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_resize = None
    max_contours = 10
    max_params = 5
    max_side_colors = 3
    layout_history_num = 60
    num_of_sides = 6
    color_layouts = np.array([[[None for i in range(max_side_colors)] for j in range(max_side_colors)] for k in range(layout_history_num)])
    color_layouts_len = 0
    final_layout_keys = {"left": 0, "front": 1, "right": 2, "back": 3, "top": 4, "bottom": 5}
    final_color_layouts = np.array([[[None for i in range(max_side_colors)] for j in range(max_side_colors)] for j in range(num_of_sides)])
    final_color_layouts_len = 0
    drawn = False

    try:
        while True:
            _, frame = cap.read()

            width = 160
            height = 120

            frame_resize = cv2.resize(frame, (width, height))
            frame_resize = cv2.flip(frame_resize, -1)
            frame_orig = frame_resize.copy()
            frame_hsv = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2HSV)
            frame_blur = cv2.GaussianBlur(frame_hsv, (5, 5), cv2.BORDER_DEFAULT)
            
            if color_layouts[color_layouts_len].all():
                if color_layouts_len == layout_history_num-1:
                    color_layouts = np.roll(color_layouts, -1, axis=0)
                    color_layouts[-1][:][:] = None
                else:
                    color_layouts_len += 1

            contour_bounding_rects = np.array([[None for i in range(max_params)] for i in range(max_contours)])
            contour_bounding_rects_len = 0
            for i, color in enumerate(colors.values()):
                contours = color.processing(frame_blur)
                for contour in contours:
                    if contour_bounding_rects_len == max_contours:
                        break
                    x_coord, y_coord, width, height = cv2.boundingRect(contour)
                    if width*height >= 75:
                        frame_resize = cv2.circle(frame_resize, (x_coord, y_coord), 3, (0, 0, 255), 2)
                        cv2.drawContours(frame_resize, [contour], -1, (0, 255, 0), 2)
                        contour_bounding_rects[contour_bounding_rects_len][:] = x_coord, y_coord, width, height, colors_keys[i]
                        contour_bounding_rects_len += 1
            
            if not (contour_bounding_rects_len == 0):
                if not (contour_bounding_rects_len == max_contours):
                    contour_bounding_rects = np.delete(contour_bounding_rects, np.argwhere(contour_bounding_rects == None)[:, 0], axis=0)
                contour_x_coords = np.sort(contour_bounding_rects[:, 0])
                contour_x_min = contour_x_coords[0]
                contour_x_max = contour_x_coords[-1]
                contour_x_max_width = contour_bounding_rects[np.argwhere(contour_x_max == contour_bounding_rects[:, 0])[0][0]][2]
                grid_x_length = contour_x_max + contour_x_max_width - contour_x_min
                contour_y_coords = np.sort(contour_bounding_rects[:, 1])
                contour_y_min = contour_y_coords[0]
                contour_y_max = contour_y_coords[-1]
                contour_y_max_height = contour_bounding_rects[np.argwhere(contour_y_max == contour_bounding_rects[:, 1])[0][0]][3]
                grid_y_length = contour_y_max + contour_y_max_height - contour_y_min
                aspect_ratio = float(grid_x_length)/float(grid_y_length)
                side = 10
                rect_x = None
                rect_y = None
                rect_x1 = None
                rect_y1 = None
                if 0.55 <= aspect_ratio <= 0.75:
                    frame_crop_right = frame_orig[max(contour_y_min, 1):(contour_y_min + grid_y_length),
                                                  min((contour_x_min + grid_x_length), frame_orig.shape[1]-1):min((contour_x_min + grid_x_length + contour_x_max_width), frame_orig.shape[1])]
                    frame_crop_left = frame_orig[max(contour_y_min, 1):(contour_y_min + grid_y_length),
                                                 max((contour_x_min - contour_x_max_width), 0):max(contour_x_min, 1)]
                    frame_right_resize = cv2.resize(frame_crop_right, (side, side))
                    frame_left_resize = cv2.resize(frame_crop_left, (side, side))
                    grid_x_length = grid_y_length
                    if np.mean(frame_right_resize[:, :, 2]) < np.mean(frame_left_resize[:, :, 2]):
                        contour_x_min -= contour_x_max_width
                        rect_x = contour_x_max + contour_x_max_width - grid_x_length
                        rect_y = contour_y_max + contour_y_max_height - grid_y_length
                        rect_x1 = contour_x_max + contour_x_max_width
                        rect_y1 = contour_y_max + contour_y_max_height
                    else:
                        rect_x = contour_x_min
                        rect_y = contour_y_min
                        rect_x1 = contour_x_min + grid_x_length
                        rect_y1 = contour_y_min + grid_y_length

                elif 1.4 <= aspect_ratio <= 1.7:
                    frame_crop_bottom = frame_orig[min((contour_y_min + grid_y_length), frame_orig.shape[0]-1):min((contour_y_min + grid_y_length + contour_y_max_height), frame_orig.shape[0]),
                                                   max(contour_x_min, 1):(contour_x_min + grid_x_length)]
                    frame_crop_top = frame_orig[max((contour_y_min - contour_y_max_height), 0):max(contour_y_min, 1),
                                                max(contour_x_min, 1):(contour_x_min + grid_x_length)]
                    frame_bottom_resize = cv2.resize(frame_crop_bottom, (side, side))
                    frame_top_resize = cv2.resize(frame_crop_top, (side, side))
                    grid_y_length = grid_x_length
                    if np.mean(frame_bottom_resize[:, :, 2]) < np.mean(frame_top_resize[:, :, 2]):
                        contour_y_min -= contour_y_max_height
                        rect_x = contour_x_max + contour_x_max_width - grid_x_length
                        rect_y = contour_y_max + contour_y_max_height - grid_y_length
                        rect_x1 = contour_x_max + contour_x_max_width
                        rect_y1 = contour_y_max + contour_y_max_height
                    else:
                        rect_x = contour_x_min
                        rect_y = contour_y_min
                        rect_x1 = contour_x_min + grid_x_length
                        rect_y1 = contour_y_min + grid_y_length
                else:
                    rect_x = contour_x_min
                    rect_y = contour_y_min
                    rect_x1 = contour_x_min + grid_x_length
                    rect_y1 = contour_y_min + grid_y_length

                frame_resize = cv2.rectangle(frame_resize, (rect_x, rect_y), (rect_x1, rect_y1),
                                             (255, 0, 0), 2)

                for bounding_rect in contour_bounding_rects:
                    axis0 = None
                    axis1 = None
                    if bounding_rect[0] > ((rect_x + rect_x1)//2):
                        axis1 = 2
                    elif bounding_rect[0] > ((rect_x + ((rect_x + rect_x1)//2))//2):
                        axis1 = 1
                    else:
                        axis1 = 0
                    if bounding_rect[1] > ((rect_y + rect_y1)//2):
                        axis0 = 2
                    elif bounding_rect[1] > ((rect_y + ((rect_y + rect_y1)//2))//2):
                        axis0 = 1
                    else:
                        axis0 = 0
                    color_layouts[color_layouts_len][axis0][axis1] = bounding_rect[4]
                color_layouts[color_layouts_len][color_layouts[color_layouts_len] == None] = colors_keys[-1]
                rgb_colors = {"orange": (0, 165, 255), "green": (0, 128, 0), "yellow": (0, 255, 255),
                              "white": (255, 255, 255), "blue": (128, 0, 0), "red": (0, 0, 128)}
                
                if (color_layouts_len == layout_history_num-1) and (final_color_layouts_len != num_of_sides):
                    first_layout = color_layouts[0]
                    if (color_layouts == first_layout).all():
                        if (final_color_layouts_len == 0) or (final_color_layouts[final_color_layouts_len-1] != first_layout).any():
                            final_color_layouts[final_color_layouts_len] = first_layout.copy()
                            final_color_layouts_len += 1
                if (final_color_layouts_len > 0) and not drawn:
                    mini_square_side = 40
                    mini_square_gap = mini_square_side//2
                    square_gap = mini_square_side//2
                    num_of_horizontal_sides = 4
                    rubix_face_length = 3*(mini_square_side + mini_square_gap)
                    synthesized_rubix_face = np.zeros((3*rubix_face_length + 2*square_gap + mini_square_gap,
                                                       4*rubix_face_length + 3*square_gap + mini_square_gap, 3), dtype=np.uint8)
                    for i in range(min(num_of_horizontal_sides, final_color_layouts_len)):
                        for j in range(len(final_color_layouts[i])):
                            for k in range(len(final_color_layouts[i][j])):
                                square_x = i*(rubix_face_length + square_gap) + mini_square_gap + k*(mini_square_side + mini_square_gap)
                                square_y = rubix_face_length + square_gap + mini_square_gap + j*(mini_square_side + mini_square_gap)
                                square_x1 = square_x + mini_square_side
                                square_y1 = square_y + mini_square_side
                                synthesized_rubix_face = cv2.rectangle(synthesized_rubix_face,
                                                                       (square_x, square_y),
                                                                       (square_x1, square_y1),
                                                                       rgb_colors[final_color_layouts[i][j][k]], -1)
                    for i in range(num_of_horizontal_sides, final_color_layouts_len):
                        for j in range(len(final_color_layouts[i])):
                            for k in range(len(final_color_layouts[i][j])):
                                square_x = rubix_face_length + square_gap + mini_square_gap + k*(mini_square_side + mini_square_gap)
                                square_y = 2*(i-4)*(rubix_face_length + square_gap) + mini_square_gap + j*(mini_square_side + mini_square_gap)
                                square_x1 = square_x + mini_square_side
                                square_y1 = square_y + mini_square_side
                                synthesized_rubix_face = cv2.rectangle(synthesized_rubix_face,
                                                                       (square_x, square_y),
                                                                       (square_x1, square_y1),
                                                                       rgb_colors[final_color_layouts[i][j][k]], -1)
                    cv2.imshow("synthesized face", synthesized_rubix_face)

            cv2.imshow(window_detection_name, cv2.resize(frame_resize, (initial_width, initial_height)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except:
        cv2.imwrite(r"/home/pi/Documents/image.jpg", frame_resize)
        cap.release()
        cv2.destroyAllWindows()
        raise KeyboardInterrupt

if __name__ == '__main__':
    main()
