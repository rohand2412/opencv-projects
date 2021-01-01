#!/usr/bin/env python3
"""This script contains all of the modules used in rubix_cube_catalogger.py"""

import numpy as np
import cv2
import my_python_modules as modules

class RubixCube(modules.ModulesPackage):
    """Class that adapts parents madules for RubixCube"""
    class Frame(modules.ModulesPackage.Frame):
        """Keeps track of all data regarding the video stream"""
        def __init__(self, name):
            super().__init__(name)
            cv2.namedWindow(self._name)

            self._resize_width = 160
            self._resize_height = 120
            self._max_indentifiable_colors = 5
            self._max_contours = 10
            self._max_contour_params = 5
            self._max_color_per_side = 3
            self._layout_history_num = 60
            self._num_of_faces = 6
            self._drawn = False

            self._color_ranges = None
            self._colors_max_values = None
            self._colors_names = None
            self._colors_keys = None
            self._colors = None
            self._color_layouts = None
            self._color_layouts_len = None
            self._final_layout_keys = None
            self._final_color_layouts = None
            self._final_color_layouts_len = None
            self._frame_orig = None
            self._frame_draw = None
            self._frame_preprocessed = None
            self._contour_bounding_rects = None
            self._contour_bounding_rects_len = None
            self._rect_x = None
            self._rect_y = None
            self._rect_x1 = None
            self._rect_y1 = None

        def init_color_data(self):
            """Initializes and stores all forms and instances of data and variables related to
            colors"""
            self._color_ranges = [[(0, 15), (130, 255), (125, 255)],
                                  [(45, 74), (137, 252), (69, 161)],
                                  [(15, 50), (75, 224), (88, 255)],
                                  [(0, 180), (0, 80), (140, 255)],
                                  [(85, 152), (110, 255), (70, 170)]]
            self._colors_max_values = [360//2, 255, 255]
            self._colors_names = ["H", "S", "V"]
            self._colors_keys = ["orange", "green", "yellow", "white", "blue", "red"]
            self._colors = {self._colors_keys[i]: \
                            modules.ModulesPackage.ColorTracker(self._colors_max_values,
                                                                self._colors_names,
                                                                self._name, self._color_ranges[i]) \
                            for i in range(self._max_indentifiable_colors)}

            self._color_layouts = np.array([[[None for i in range(self._max_color_per_side)] \
                                             for j in range(self._max_color_per_side)] \
                                            for k in range(self._layout_history_num)])
            self._color_layouts_len = 0
            self._final_layout_keys = {"left": 0, "front": 1, "right": 2, "back": 3, "top": 4,
                                       "bottom": 5}
            self._final_color_layouts = np.array([[[None for i in range(self._max_color_per_side)] \
                                                   for j in range(self._max_color_per_side)] \
                                                  for j in range(self._num_of_faces)])
            self._final_color_layouts_len = 0

        def preprocessing(self):
            """Preprocesses the frame"""
            frame_resize = cv2.resize(self._frame, (self._resize_width, self._resize_height))
            frame_resize = cv2.flip(frame_resize, -1)
            frame_hsv = cv2.cvtColor(frame_resize.copy(), cv2.COLOR_BGR2HSV)
            frame_blur = cv2.GaussianBlur(frame_hsv, (5, 5), cv2.BORDER_DEFAULT)
            self._frame_orig = frame_resize.copy()
            self._frame_draw = self._frame_orig.copy()
            self._frame_preprocessed = frame_blur.copy()

        def find_contours(self):
            """Finds the contours in the ranges specified earlier and stores them for later use"""
            if self._color_layouts[self._color_layouts_len].all():
                if self._color_layouts_len == self._layout_history_num-1:
                    self._color_layouts = np.roll(self._color_layouts, -1, axis=0)
                    self._color_layouts[-1][:][:] = None
                else:
                    self._color_layouts_len += 1

            self._contour_bounding_rects = np.array([[None for i in range(self._max_contour_params)] \
                                                     for i in range(self._max_contours)])
            self._contour_bounding_rects_len = 0
            for i, color in enumerate(self._colors.values()):
                contours = color.processing(self._frame_preprocessed)
                for contour in contours:
                    if self._contour_bounding_rects_len == self._max_contours:
                        break
                    x_coord, y_coord, width, height = cv2.boundingRect(contour)
                    if width*height >= 75:
                        self._frame_draw = cv2.circle(self._frame_draw, (x_coord, y_coord), 3,
                                                      (0, 0, 255), 2)
                        self._frame_draw = cv2.drawContours(self._frame_draw, [contour], -1,
                                                            (0, 255, 0), 2)
                        self._contour_bounding_rects[self._contour_bounding_rects_len][:] = \
                                        x_coord, y_coord, width, height, self._colors_keys[i]
                        self._contour_bounding_rects_len += 1

        def processing(self):
            """Processes the frame"""
            if self._contour_bounding_rects_len != 0:
                self._accomodate_red_strips()
                self._log_color_layout()
                self._log_final_color_layout()
                self._plot()

        def _accomodate_red_strips(self):
            """Accomodates the cube face bounding box for red stripes since we aren't indentifying 
            red squares"""
            if self._contour_bounding_rects_len != self._max_contours:
                self._contour_bounding_rects = np.delete(self._contour_bounding_rects,
                                                         np.argwhere(self._contour_bounding_rects == None)[:, 0],
                                                         axis=0)
            contour_x_coords = np.sort(self._contour_bounding_rects[:, 0])
            contour_x_min = contour_x_coords[0]
            contour_x_max = contour_x_coords[-1]
            contour_x_max_width = self._contour_bounding_rects[np.argwhere(contour_x_max == self._contour_bounding_rects[:, 0])[0][0]][2]
            grid_x_length = contour_x_max + contour_x_max_width - contour_x_min
            contour_y_coords = np.sort(self._contour_bounding_rects[:, 1])
            contour_y_min = contour_y_coords[0]
            contour_y_max = contour_y_coords[-1]
            contour_y_max_height = self._contour_bounding_rects[np.argwhere(contour_y_max == self._contour_bounding_rects[:, 1])[0][0]][3]
            grid_y_length = contour_y_max + contour_y_max_height - contour_y_min
            aspect_ratio = float(grid_x_length)/float(grid_y_length)
            side = 10
            if 0.55 <= aspect_ratio <= 0.75:
                frame_crop_right = self._frame_orig[max(contour_y_min, 1):(contour_y_min + grid_y_length),
                                                    min((contour_x_min + grid_x_length),
                                                        self._frame_orig.shape[1]-1):
                                                    min((contour_x_min + grid_x_length + contour_x_max_width),
                                                        self._frame_orig.shape[1])]
                frame_crop_left = self._frame_orig[max(contour_y_min, 1):(contour_y_min + grid_y_length),
                                                   max((contour_x_min - contour_x_max_width), 0):
                                                   max(contour_x_min, 1)]
                frame_right_resize = cv2.resize(frame_crop_right, (side, side))
                frame_left_resize = cv2.resize(frame_crop_left, (side, side))
                grid_x_length = grid_y_length
                if np.mean(frame_right_resize[:, :, 2]) < np.mean(frame_left_resize[:, :, 2]):
                    contour_x_min -= contour_x_max_width
                    self._rect_x = contour_x_max + contour_x_max_width - grid_x_length
                    self._rect_y = contour_y_max + contour_y_max_height - grid_y_length
                    self._rect_x1 = contour_x_max + contour_x_max_width
                    self._rect_y1 = contour_y_max + contour_y_max_height
                else:
                    self._rect_x = contour_x_min
                    self._rect_y = contour_y_min
                    self._rect_x1 = contour_x_min + grid_x_length
                    self._rect_y1 = contour_y_min + grid_y_length

            elif 1.4 <= aspect_ratio <= 1.7:
                frame_crop_bottom = self._frame_orig[min((contour_y_min + grid_y_length),
                                                         self._frame_orig.shape[0]-1):
                                                     min((contour_y_min + grid_y_length + contour_y_max_height), 
                                                         self._frame_orig.shape[0]),
                                                     max(contour_x_min, 1):(contour_x_min + grid_x_length)]
                frame_crop_top = self._frame_orig[max((contour_y_min - contour_y_max_height), 0):
                                                  max(contour_y_min, 1),
                                                  max(contour_x_min, 1):(contour_x_min + grid_x_length)]
                frame_bottom_resize = cv2.resize(frame_crop_bottom, (side, side))
                frame_top_resize = cv2.resize(frame_crop_top, (side, side))
                grid_y_length = grid_x_length
                if np.mean(frame_bottom_resize[:, :, 2]) < np.mean(frame_top_resize[:, :, 2]):
                    contour_y_min -= contour_y_max_height
                    self._rect_x = contour_x_max + contour_x_max_width - grid_x_length
                    self._rect_y = contour_y_max + contour_y_max_height - grid_y_length
                    self._rect_x1 = contour_x_max + contour_x_max_width
                    self._rect_y1 = contour_y_max + contour_y_max_height
                else:
                    self._rect_x = contour_x_min
                    self._rect_y = contour_y_min
                    self._rect_x1 = contour_x_min + grid_x_length
                    self._rect_y1 = contour_y_min + grid_y_length
            else:
                self._rect_x = contour_x_min
                self._rect_y = contour_y_min
                self._rect_x1 = contour_x_min + grid_x_length
                self._rect_y1 = contour_y_min + grid_y_length

            self._frame_draw = cv2.rectangle(self._frame_draw, (self._rect_x, self._rect_y), 
                                             (self._rect_x1, self._rect_y1), (255, 0, 0), 2)

        def _log_color_layout(self):
            """Logs the color layout and stores it in history buffer"""
            for bounding_rect in self._contour_bounding_rects:
                axis0 = None
                axis1 = None
                if bounding_rect[0] > ((self._rect_x + self._rect_x1)//2):
                    axis1 = 2
                elif bounding_rect[0] > ((self._rect_x + ((self._rect_x + self._rect_x1)//2))//2):
                    axis1 = 1
                else:
                    axis1 = 0
                if bounding_rect[1] > ((self._rect_y + self._rect_y1)//2):
                    axis0 = 2
                elif bounding_rect[1] > ((self._rect_y + ((self._rect_y + self._rect_y1)//2))//2):
                    axis0 = 1
                else:
                    axis0 = 0
                self._color_layouts[self._color_layouts_len][axis0][axis1] = bounding_rect[4]
            self._color_layouts[self._color_layouts_len][self._color_layouts[self._color_layouts_len] == None] = self._colors_keys[-1]

        def _log_final_color_layout(self):
            """Logs the final color layout and stores it in list of final color layouts of the other 
            faces"""
            if (self._color_layouts_len == self._layout_history_num-1) and (self._final_color_layouts_len != self._num_of_faces):
                first_layout = self._color_layouts[0]
                if (self._color_layouts == first_layout).all():
                    if (self._final_color_layouts_len == 0) or (self._final_color_layouts[self._final_color_layouts_len-1] != first_layout).any():
                        self._final_color_layouts[self._final_color_layouts_len] = first_layout.copy()
                        self._final_color_layouts_len += 1

        def _plot(self):
            """Plots the final determined faces of the rubix cube, does not need all six faces to 
            begin plotting"""
            rgb_colors = {"orange": (0, 165, 255), "green": (0, 128, 0), "yellow": (0, 255, 255),
                          "white": (255, 255, 255), "blue": (128, 0, 0), "red": (0, 0, 128)}
            if (self._final_color_layouts_len > 0) and not self._drawn:
                mini_square_side = 40
                mini_square_gap = mini_square_side//2
                square_gap = mini_square_side//2
                num_of_horizontal_sides = 4
                rubix_face_length = 3*(mini_square_side + mini_square_gap)
                synthesized_rubix_face = np.zeros((3*rubix_face_length + 2*square_gap + mini_square_gap,
                                                   4*rubix_face_length + 3*square_gap + mini_square_gap, 3), dtype=np.uint8)
                for i in range(min(num_of_horizontal_sides, self._final_color_layouts_len)):
                    for j in range(len(self._final_color_layouts[i])):
                        for k in range(len(self._final_color_layouts[i][j])):
                            square_x = i*(rubix_face_length + square_gap) + mini_square_gap + k*(mini_square_side + mini_square_gap)
                            square_y = rubix_face_length + square_gap + mini_square_gap + j*(mini_square_side + mini_square_gap)
                            square_x1 = square_x + mini_square_side
                            square_y1 = square_y + mini_square_side
                            synthesized_rubix_face = cv2.rectangle(synthesized_rubix_face,
                                                                   (square_x, square_y),
                                                                   (square_x1, square_y1),
                                                                   rgb_colors[self._final_color_layouts[i][j][k]], -1)
                for i in range(num_of_horizontal_sides, self._final_color_layouts_len):
                    for j in range(len(self._final_color_layouts[i])):
                        for k in range(len(self._final_color_layouts[i][j])):
                            square_x = rubix_face_length + square_gap + mini_square_gap + k*(mini_square_side + mini_square_gap)
                            square_y = 2*(i-4)*(rubix_face_length + square_gap) + mini_square_gap + j*(mini_square_side + mini_square_gap)
                            square_x1 = square_x + mini_square_side
                            square_y1 = square_y + mini_square_side
                            synthesized_rubix_face = cv2.rectangle(synthesized_rubix_face,
                                                                   (square_x, square_y),
                                                                   (square_x1, square_y1),
                                                                   rgb_colors[self._final_color_layouts[i][j][k]], -1)
                cv2.imshow("synthesized face", synthesized_rubix_face)

        def imshow(self):
            """Displays the frame and resizes it for convenience"""
            cv2.imshow(self._name, cv2.resize(self._frame_draw, (self._width, self._height)))
