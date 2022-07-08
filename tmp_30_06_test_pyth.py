#!/usr/bin/env python
# coding: utf-8
import time
import cv2
import tensorflow as tf
from object_detection.utils import config_util
import collections
from object_detection.utils import visualization_utils as viz_utils
import six
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.Image as Image
import os
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
import numpy as np
#from object_detection.protos import pipeline_pb2
#from google.protobuf import text_format


# # Object classes

class Hochregallager:
    def __init__(self):
        self.behaelter_obj_list = []
        self.width_in_px = None
        self.height_in_px = None
        self.coordinates = None
        self.behaelter_arr = [[None for x in range(3)] for y in range(3)]
        self.grid_cell_timer_arr = [[0 for x in range(3)] for y in range(3)]
        self.grid_successfully_initialized = False

    def add_behaelter(self, behaelter_obj):
        self.behaelter_obj_list.append(behaelter_obj)

    def assign_grid_pos(self, behaelter, row, column):
        self.behaelter_arr[row][column] = behaelter
        if behaelter.empty:
            print('POS:{}x{} is EMPTY')
        else:
            print('POS:{}x{} is {}'.format(
                row, column, behaelter.werk_stueck.color))

    def check_for_missing_behaelter(self):
        # check if a behaelter was removed, if so - call remove
        pass

    def remove_behaelter(self, row, column):
        self.behaelter_arr[row][column] = None
        # -> TIMER

    # call after a new frame is presented

    def clear_behaelter_list(self):
        self.behaelter_obj_list = []

    def initialize_grid_coordinates(self):
        # coordinates should only be None before the first frame, grid_successfully_initialized being false
        # means (at least) one Behaelter was missing in the frame(s) before, thereby distorting the coordinates
        if not self.grid_successfully_initialized:
            self.coordinates = get_approx_hochregallager_grid_coordinates(
                hochregallager)
            self.width_in_px = get_approx_hochregallager_grid_width(
                hochregallager)
            self.height_in_px = get_approx_hochregallager_grid_height(
                hochregallager)
            if len(self.behaelter_obj_list) == 9:
                self.grid_successfully_initialized = True

    def start_grid_cell_timer(self, row, column, current_time):
        self.grid_cell_timer_arr[row][column] = current_time
        print("TIMER WAS STARTED")

    def stop_grid_cell_timer(self, row, column):
        self.grid_cell_timer_arr[row][column] = 0

    def get_behaelter_pos_by_behaelter_obj(self, behaelter_obj):
        # grid shape: 3x3
        for row in range(3):
            for column in range(3):
                if behaelter_obj == self.behaelter_arr[row][column]:
                    return (row, column)
        return None

    def get_behaelter_pos_by_behaelter_box(self, image, behaelter_box):
        im_height = image.shape[0]
        im_width = image.shape[1]
        # behaelter_arr contains behaelter obj with de'normalized' coordinates
        behaelter_box = get_box_coordinates_from_normalized_coordinates(
            behaelter_box, im_height, im_width)
        # grid shape: 3x3
        for row in range(3):
            for column in range(3):
                # REMOVE following line
                if self.behaelter_arr[row][column] is None:
                    pass

                else:
                    # create a numpy 2d arr (needed for bounding_box_intersect)
                    #tmp_np_arr = np.array([self.behaelter_arr[row][column].bounding_box])

                    # since behaelter obj in hochregallager.behaelter_arr only get removed after they disappear, bounding boxes
                    # of behaelter in newer frames wont fit the previous (or possibly first) behaelter coords, 0.8 percent
                    # should be adequate
                    # intersect_element = bounding_box_intersect(
                    #    behaelter_box,
                    #    tmp_np_arr,
                    #    im_height=im_height,
                    #    im_width=im_width,
                    #    needs_normalization=False,
                    #    return_percent=False,
                    #    percent_threshold=0.8
                    # )

                    # if len(intersect_element) == 1:
                    #    return (row,column)

                    if behaelter_box == self.behaelter_arr[row][column].bounding_box:
                        return (row, column)

        return None

    def get_behaelter_obj_by_behaelter_box(self, image, behaelter_box):
        im_height = image.shape[0]
        im_width = image.shape[1]
        # behaelter_arr contains behaelter obj with de'normalized' coordinates
        behaelter_box = get_box_coordinates_from_normalized_coordinates(
            behaelter_box, im_height, im_width)
        # grid shape: 3x3
        for behaelter_obj in self.behaelter_obj_list:
            if behaelter_obj.bounding_box == behaelter_box:
                return behaelter_obj
        return None

    def get_werkstueck_pos_by_werkstueck_box(self, image, wkstk_box):
        im_height = image.shape[0]
        im_width = image.shape[1]
        # behaelter_arr contains behaelter obj with de'normalized' coordinates
        wkstk_box = get_box_coordinates_from_normalized_coordinates(
            wkstk_box, im_height, im_width)
        # grid shape: 3x3
        for row in range(3):
            for column in range(3):
                if wkstk_box == self.behaelter_arr[row][column].werk_stueck.bounding_box:
                    return (row, column)
        return None


class WerkStueck:
    def __init__(self, box, score, color):

        # ymin, xmin, ymax, xmax = box
        self.bounding_box = box
        self.score = score
        self.color = color


class Behaelter:
    def __init__(self, image, behaelter_box, behaelter_score, filtered_WerkStueck_detections):
        im_height = image.shape[0]
        im_width = image.shape[1]

        wkst_boxes, _, wkst_scores = filtered_WerkStueck_detections

        # ymin, xmin, ymax, xmax
        self.bounding_box = get_box_coordinates_from_normalized_coordinates(
            behaelter_box, im_height, im_width)
        self.werk_stueck = self.has_WerkStueck(
            wkst_boxes, wkst_scores, image, im_height, im_width)
        self.empty = True if self.werk_stueck is None else False

    def has_WerkStueck(self, wkst_boxes, wkst_scores, image, im_height, im_width):
        return get_werkstueck_contained_in_behaelter(self.bounding_box, wkst_boxes, wkst_scores, image, im_height, im_width)


# # Intersect methods

def get_box_coordinates_from_normalized_coordinates(
    norm_box, im_height, im_width, return_int=False
):
    ymin, xmin, ymax, xmax = (
        norm_box[0] * im_height,
        norm_box[1] * im_width,
        norm_box[2] * im_height,
        norm_box[3] * im_width,
    )
    if return_int:
        ymin = int(ymin)
        xmin = int(xmin)
        ymax = int(ymax)
        xmax = int(xmax)
    return (ymin, xmin, ymax, xmax)


def one_dim_intersect(a0, a1, b0, b1):
    # a contains b
    if a0 < b0 and a1 > b1:
        intersect = b1 - b0
    # b contains a
    elif a0 >= b0 and a1 <= b1:
        intersect = a1 - a0
    # right part of a and left part of b intersect
    elif a0 < b0 and a1 > b0:
        intersect = a1 - b0
    # left part of a and right part of b intersect
    elif a1 > b1 and a0 < b1:
        intersect = b1 - a0
    # no intersect
    else:
        intersect = 0
    return intersect


def get_werkstueck_contained_in_behaelter(
    behaelter_box, wkst_boxes, wkst_scores, image, im_height, im_width
):
    intersect_elements = bounding_box_intersect(
        behaelter_box, wkst_boxes, wkst_scores, im_height=im_height, im_width=im_width, needs_normalization=True)

    if len(intersect_elements) == 1:
        wkstk_box = intersect_elements[0][0]
        wkstk_score = intersect_elements[0][1]
        wkstk_color = detect_color_in_bounding_box(image, wkstk_box, False)

        wkstk = WerkStueck(wkstk_box, wkstk_score, wkstk_color)
        return wkstk

    else:
        if len(intersect_elements) > 1:
            print(
                "{} wkstk elements intersect behaelter:{}".format(
                    len(intersect_elements), behaelter_box
                )
            )
        return None


def bounding_box_intersect(
    primary_box, secondary_boxes, secondary_boxes_scores=None, im_height=None, im_width=None, needs_normalization=True, return_percent=False, percent_threshold=0.2
):
    p_ymin, p_xmin, p_ymax, p_xmax = primary_box
    primary_box_area = float((p_xmax - p_xmin) * (p_ymax - p_ymin))
    # using list to check for errors, only one elem should be appended
    intersect_elements = []

    # loop over all WerkStueck bounding boxes and find the corresponding one
    for i in range(secondary_boxes.shape[0]):
        if needs_normalization:
            (
                s_ymin,
                s_xmin,
                s_ymax,
                s_xmax,
            ) = get_box_coordinates_from_normalized_coordinates(
                secondary_boxes[i], im_height, im_width
            )
        else:
            (
                s_ymin,
                s_xmin,
                s_ymax,
                s_xmax,
            ) = secondary_boxes[i]

        intersect_width = one_dim_intersect(p_xmin, p_xmax, s_xmin, s_xmax)
        intersect_height = one_dim_intersect(p_ymin, p_ymax, s_ymin, s_ymax)
        intersect_area = intersect_width * intersect_height
        intersect_percent = intersect_area / primary_box_area
        # print(intersect_percent)
        if intersect_percent >= percent_threshold:
            if secondary_boxes_scores is None:
                if return_percent:
                    intersect_elements.append(
                        ((s_ymin, s_xmin, s_ymax, s_xmax), intersect_percent)
                    )
                else:
                    intersect_elements.append(
                        (s_ymin, s_xmin, s_ymax, s_xmax)
                    )
            else:
                if return_percent:
                    intersect_elements.append(
                        ((s_ymin, s_xmin, s_ymax, s_xmax),
                         secondary_boxes_scores[i], intersect_percent)
                    )
                else:
                    intersect_elements.append(
                        ((s_ymin, s_xmin, s_ymax, s_xmax),
                         secondary_boxes_scores[i])
                    )

    return intersect_elements


# # Grid location & timer start/stop

# get coordinates for lowest ymin xmin, highest ymax xmax to find the 4 corners
# should only be called when hochregallager.behaelter_obj_list has AT LEAST one behaelter
def get_approx_hochregallager_grid_coordinates(hochregallager):
    behaelter_list = hochregallager.behaelter_obj_list

    ymin_test, xmin_test, ymax_test, xmax_test = behaelter_list[0].bounding_box[0], behaelter_list[
        0].bounding_box[1], behaelter_list[0].bounding_box[2], behaelter_list[0].bounding_box[3]

    for behaelter in behaelter_list:
        print(type(behaelter))
        new_ymin, new_xmin, new_ymax, new_xmax = behaelter.bounding_box
        if ymin_test > new_ymin:
            ymin_test = new_ymin
        if xmin_test > new_xmin:
            xmin_test = new_xmin
        if ymax_test < new_ymax:
            ymax_test = new_ymax
        if xmax_test < new_xmax:
            xmax_test = new_xmax

        # REMOVE
        #image_np_for_grid = image_np_with_detections.copy()
        #start_point = (int(new_xmin), int(new_ymin))
        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        #end_point = (int(new_xmax), int(new_ymax))
        # Blue color in BGR
        #color = (255, 255, 0)
        # Line thickness of 2 px
        #thickness = 4
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        #cv2.rectangle(image_np_with_detections, start_point, end_point, color, thickness)
        #cv2.polylines(image_np_with_detections, (top, left, bottom, right), True, (0, 0, 0), 4)
        #cv2.imshow("object detection", cv2.resize(image_np_for_grid, (1500, 1200)))
        # cv2.waitKey(0)

    return (ymin_test, xmin_test, ymax_test, xmax_test)


def get_approx_hochregallager_grid_width(hochregallager):
    _, xmin, _, xmax = hochregallager.coordinates
    return xmax-xmin


def get_approx_hochregallager_grid_height(hochregallager):
    ymin, _, ymax, _ = hochregallager.coordinates
    return ymax-ymin


def assign_grid_positions(hochregallager):
    ymin, xmin, _, _ = hochregallager.coordinates
    grid_width_in_px, grid_height_in_px = hochregallager.width_in_px, hochregallager.height_in_px

    # REMOVE
    start_point = (int(hochregallager.coordinates[1]), int(
        hochregallager.coordinates[0]))
    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (int(hochregallager.coordinates[3]), int(
        hochregallager.coordinates[2]))
    # Blue color in BGR
    color = (255, 255, 0)
    # Line thickness of 2 px
    thickness = 4
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    cv2.rectangle(image_np_with_detections, start_point,
                  end_point, color, thickness)
    #cv2.polylines(image_np_with_detections, (top, left, bottom, right), True, (0, 0, 0), 4)

    # create a numpy 2d arr for bounding boxes of hochregallager.behaelter_obj_list (needed for bounding_box_intersect)
    tmp_list = []
    for obj in hochregallager.behaelter_obj_list:
        tmp_list.append(obj.bounding_box)
    behaelter_np_arr = np.array(tmp_list)

    percent = 1/3
    grid_cell_width = grid_width_in_px*percent
    grid_cell_height = grid_height_in_px*percent

    top = ymin - grid_cell_height
    bottom = ymin

    # 3x3 hochregallager array
    # loop over all POSs in hochregallager.behaelter_arr and change status accordingly
    for i in range(3):
        top = top + (grid_cell_height)
        bottom = bottom + (grid_cell_height)
        left, right = (xmin - grid_cell_width), xmin
        for j in range(3):
            left = left + (grid_cell_width)
            right = right + (grid_cell_width)

            # REMOVE
            #start_point = (int(left), int(top))
            # Ending coordinate, here (220, 220)
            # represents the bottom right corner of rectangle
            #end_point = (int(right), int(bottom))
            # Blue color in BGR
            #color = (255, 255, 0)
            # Line thickness of 2 px
            #thickness = 4
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            #cv2.rectangle(image_np_with_detections, start_point, end_point, color, thickness)
            #cv2.polylines(image_np_with_detections, (top, left, bottom, right), True, (0, 0, 0), 4)
            #cv2.imshow("object detection", cv2.resize(image_np_with_detections, (1500, 1200)))
            # cv2.waitKey(0)

            grid_cell_bounding_box = (top, left, bottom, right)
            intersect_elements = bounding_box_intersect(
                grid_cell_bounding_box, behaelter_np_arr, needs_normalization=False, return_percent=True)

            if len(intersect_elements) >= 1:
                # if more than one behaelter is (partially) in a grid cell, then get the behaelter with the highest overlap
                behaelter_obj = None
                actual_intersect_elem = intersect_elements[0]
                if len(intersect_elements) > 1:
                    highest_intersect_percent = intersect_elements[0][1]
                    for elem in intersect_elements:
                        if elem[1] > highest_intersect_percent:
                            highest_intersect_percent = elem[1]
                            actual_intersect_elem = elem
                # find corresponding behaelter object and assign to grid
                for elem in hochregallager.behaelter_obj_list:
                    if elem.bounding_box == actual_intersect_elem[0]:
                        behaelter_obj = elem

                # grid cell is filled and was empty
                if hochregallager.behaelter_arr[i][j] is None:
                    hochregallager.assign_grid_pos(
                        behaelter_obj, row=i, column=j)
                    hochregallager.stop_grid_cell_timer(row=i, column=j)
                # grid cell is filled and was filled before
                else:
                    # update behaelter obj, possibly changing wkstk color and/or scores
                    hochregallager.assign_grid_pos(
                        behaelter_obj, row=i, column=j)

            # no behaelter found in grid cell (len(intersect_elements) == 0)
            else:
                print('No behaelter found, POS:{}x{}'.format(i, j))
                # grid cell is empty and was empty
                if hochregallager.behaelter_arr[i][j] is None:
                    pass
                # grid cell is empty and was filled before
                else:
                    hochregallager.remove_behaelter(row=i, column=j)
                    current_time = time.time()
                    hochregallager.start_grid_cell_timer(
                        row=i, column=j, current_time=current_time)


def get_box_coord_relative_to_grid_coord(image, box, hochregallager):
    im_height = image.shape[0]
    im_width = image.shape[1]
    h_ymin, h_xmin, _, _ = hochregallager.coordinates
    b_ymin, b_xmin, b_ymax, b_xmax = get_box_coordinates_from_normalized_coordinates(
        box, im_height, im_width)
    relative_ymin = b_ymin - h_ymin
    relative_xmin = b_xmin - h_xmin
    relative_ymax = b_ymax - h_ymin
    relative_xmax = b_xmax - h_xmin
    return (relative_ymin, relative_xmin, relative_ymax, relative_xmax)


# # Timer for missing Behaelter

# In[7]:


# timer runtime in sec
def get_grid_cell_timer_value(row, column):
    time_diff = time.time() - hochregallager.grid_cell_timer_arr[row][column]
    return round(time_diff, 2)


# # Setup

WORKSPACE_PATH = "Tensorflow/workspace"
SCRIPTS_PATH = "Tensorflow/scripts"
APIMODEL_PATH = "Tensorflow/models"
ANNOTATION_PATH = WORKSPACE_PATH + "/annotations"
IMAGE_PATH = WORKSPACE_PATH + "/images"
MODEL_PATH = WORKSPACE_PATH + "/models"
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + "/pre-trained-models"
CONFIG_PATH = MODEL_PATH + "/my_ssd_mobnet/pipeline.config"
CHECKPOINT_PATH = MODEL_PATH + "/my_ssd_mobnet/"
CUSTOM_MODEL_NAME = "my_ssd_mobnet"

CONFIG_PATH = MODEL_PATH + "/" + CUSTOM_MODEL_NAME + "/pipeline.config"


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(
    model_config=configs["model"], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, "ckpt-21")).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# # Color detection

# COLOR DETECT - get color for single bounding box


# COLOR THRESHOLDS
# 110 lower bounds to make sure white(/black) isn't mistakenly recognized
lower_sat = 110
lower_val = 110
upper_sat = 255
upper_val = 255

# RED
red_mask_lower_1 = np.array([0, lower_sat, lower_val], np.uint8)
red_mask_upper_1 = np.array([15, upper_sat, upper_val], np.uint8)
red_mask_lower_2 = np.array([165, lower_sat, lower_val], np.uint8)
red_mask_upper_2 = np.array([179, upper_sat, upper_val], np.uint8)

# BLUE
blue_mask_lower = np.array([105, lower_sat, lower_val], np.uint8)
blue_mask_upper = np.array([125, upper_sat, upper_val], np.uint8)

# WHITE (sat 0-110, all other colors start at 110; val 175-255 removes noise)
white_mask_lower = np.array([0, 0, 175], np.uint8)
white_mask_upper = np.array([179, lower_sat, upper_val], np.uint8)

# GREEN
green_mask_lower = np.array([45, lower_sat, lower_val], np.uint8)
green_mask_upper = np.array([85, upper_sat, upper_val], np.uint8)

# ORANGE
orange_mask_lower = np.array([15, lower_sat, lower_val], np.uint8)
orange_mask_upper = np.array([22, upper_sat, upper_val], np.uint8)

# YELLOW
yellow_mask_lower = np.array([22, lower_sat, lower_val], np.uint8)
yellow_mask_upper = np.array([33, upper_sat, upper_val], np.uint8)

# CYAN
cyan_mask_lower = np.array([85, lower_sat, lower_val], np.uint8)
cyan_mask_upper = np.array([105, upper_sat, upper_val], np.uint8)

# VIOLET
violet_mask_lower = np.array([125, lower_sat, lower_val], np.uint8)
violet_mask_upper = np.array([140, upper_sat, upper_val], np.uint8)

# MAGENTA
magenta_mask_lower = np.array([140, lower_sat, lower_val], np.uint8)
magenta_mask_upper = np.array([155, upper_sat, upper_val], np.uint8)

# PURPLE
purple_mask_lower = np.array([155, lower_sat, lower_val], np.uint8)
purple_mask_upper = np.array([165, upper_sat, upper_val], np.uint8)


def detect_color_in_bounding_box(image_np, current_box, used_normalized_coordinates):
    image = image_np.copy()
    top, left, bottom, right = current_box
    # adjust coordinates:
    if used_normalized_coordinates:
        (top, left, bottom, right) = get_box_coordinates_from_normalized_coordinates(
            current_box, image.shape[0], image.shape[1]
        )

    # coordinates corresponding to the center 50% of the bounding box surface area
    y = int(top + (bottom - top) * 0.25)
    h = int(top + (bottom - top) * 0.75)
    x = int(left + (right - left) * 0.25)
    w = int(left + (right - left) * 0.75)

    # crop image to center of current bounding box
    cropped_img = image[y:h, x:w]

    # TRANSFORMATION to remove noise
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(cropped_img, kernel, iterations=2)
    img_dilated_and_eroded = cv2.erode(img_dilation, kernel, iterations=2)

    # HSV based frame
    hsv_frame = cv2.cvtColor(img_dilated_and_eroded, cv2.COLOR_BGR2HSV)

    # MASKS
    red_mask_1 = cv2.inRange(hsv_frame, red_mask_lower_1, red_mask_upper_1)
    red_mask_2 = cv2.inRange(hsv_frame, red_mask_lower_2, red_mask_upper_2)
    blue_mask = cv2.inRange(hsv_frame, blue_mask_lower, blue_mask_upper)
    white_mask = cv2.inRange(hsv_frame, white_mask_lower, white_mask_upper)
    green_mask = cv2.inRange(hsv_frame, green_mask_lower, green_mask_upper)
    orange_mask = cv2.inRange(hsv_frame, orange_mask_lower, orange_mask_upper)
    yellow_mask = cv2.inRange(hsv_frame, yellow_mask_lower, yellow_mask_upper)
    cyan_mask = cv2.inRange(hsv_frame, cyan_mask_lower, cyan_mask_upper)
    violet_mask = cv2.inRange(hsv_frame, violet_mask_lower, violet_mask_upper)
    magenta_mask = cv2.inRange(
        hsv_frame, magenta_mask_lower, magenta_mask_upper)
    purple_mask = cv2.inRange(hsv_frame, purple_mask_lower, purple_mask_upper)

    masks = [
        (red_mask_1, "RED"),
        (red_mask_2, "RED"),
        (blue_mask, "BLUE"),
        (white_mask, "WHITE"),
        (green_mask, "GREEN"),
        (orange_mask, "ORANGE"),
        (yellow_mask, "YELLOW"),
        (cyan_mask, "CYAN"),
        (violet_mask, "VIOLET"),
        (magenta_mask, "MAGENTA"),
        (purple_mask, "PURPLE"),
    ]

    # loop over masks and get corresponding color for bounding box
    object_color = "UNDECIDED"
    for mask in masks:
        if np.count_nonzero(mask[0]) > (mask[0].size * 0.5):
            object_color = mask[1]

    return object_color


# # Filter methods

# In[15]:


# params
#boxes = detections['detection_boxes']
#classes = detections['detection_classes']+label_id_offset
#scores = detections['detection_scores']
def filter_detections_by_score(boxes, classes, scores, min_score_thresh):

    # Create an empty (boolean) list for filtering detections
    filter_arr = []

    # boxes: [N,4], N = number of boxes
    for i in range(boxes.shape[0]):
        if scores[i] > min_score_thresh:
            filter_arr.append(True)
        else:
            filter_arr.append(False)

    # detections above min score threshold
    filtered_boxes = boxes[filter_arr]
    filtered_classes = classes[filter_arr]
    filtered_scores = scores[filter_arr]

    return (filtered_boxes, filtered_classes, filtered_scores)


# params
#boxes = detections['detection_boxes']
#classes = detections['detection_classes']+label_id_offset
#scores = detections['detection_scores']
def filter_detections_by_class(boxes, classes, scores, category_index):

    # Create an empty (boolean) list for filtering detections
    filter_arr_WerkStueck = []

    # boxes: [N,4], N = number of boxes
    for i in range(boxes.shape[0]):
        if category_index[classes[i]]['name'] == "WerkStueck":
            filter_arr_WerkStueck.append(True)
        else:
            filter_arr_WerkStueck.append(False)

    # flip WerkStueck boolean filter list
    filter_arr_Behaelter = [not elem for elem in filter_arr_WerkStueck]

    filtered_WerkStueck_boxes = boxes[filter_arr_WerkStueck]
    filtered_WerkStueck_classes = classes[filter_arr_WerkStueck]
    filtered_WerkStueck_scores = scores[filter_arr_WerkStueck]

    filtered_Behaelter_boxes = boxes[filter_arr_Behaelter]
    filtered_Behaelter_classes = classes[filter_arr_Behaelter]
    filtered_Behaelter_scores = scores[filter_arr_Behaelter]

    filtered_WerkStueck_detections = (
        filtered_WerkStueck_boxes, filtered_WerkStueck_classes, filtered_WerkStueck_scores)
    filtered_Behaelter_detections = (
        filtered_Behaelter_boxes, filtered_Behaelter_classes, filtered_Behaelter_scores)

    return (filtered_WerkStueck_detections, filtered_Behaelter_detections)


# # Visualization
# @override visualization_utils.visualize_boxes_and_labels_on_image_array
# Modified version - removed all unnecessary parts

# In[49]:


STANDARD_COLORS = viz_utils.STANDARD_COLORS


def visualize_boxes_and_labels_for_behaelter_and_werkstueck(
    image,
    boxes,
    classes,
    scores,
    category_index,
    visualize_werkstueck,
    behaelter_detections,
    hochregallager,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=0.5,
    agnostic_mode=False,
    line_thickness=4,
    mask_alpha=0.4,
    groundtruth_box_visualization_color="black",
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
):
    """Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    visualize_werkstueck: whether to skip visualizing WerkStueck class
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box or keypoint to be
      visualized.
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    mask_alpha: transparency value between 0 and 1 (default: 0.4).
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_boxes: whether to skip the drawing of bounding boxes.
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # put timer text for missing behaelter
    visualize_missing_behaelter_timer(image, hochregallager)

    # by default only visualize 'Behaelter'
    if not visualize_werkstueck:
        boxes, classes, scores = behaelter_detections

    box_to_display_str_map = collections.defaultdict(list)
    str_list = []
    box_to_color_map = collections.defaultdict(str)
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        box = tuple(boxes[i].tolist())
        if scores is None:
            box_to_color_map[box] = groundtruth_box_visualization_color
        else:
            display_str = ""
            display_str2 = ""
            if not skip_labels:
                if not agnostic_mode:
                    if classes[i] in six.viewkeys(category_index):
                        class_name = category_index[classes[i]]["name"]
                    else:
                        class_name = "N/A"
                    display_str = str(class_name)
                ################
                if category_index[classes[i]]["name"] == "Behaelter":
                    pos = hochregallager.get_behaelter_pos_by_behaelter_box(
                        image, box)
                    print("FIRST pos: {}".format(pos))
                    behaelter = hochregallager.get_behaelter_obj_by_behaelter_box(
                        image, box)
                    if pos is None:
                        display_str2 = "POS: N/A"
                    else:
                        row, column = pos
                        display_str2 = "POS: {}x{}".format(row, column)

                    ymin, xmin, ymax, xmax = get_box_coord_relative_to_grid_coord(
                        image, box, hochregallager)
                    display_str2 = "{}, Left:{} Right:{} Top:{} Bottom:{}".format(
                        display_str2, round(xmin), round(xmax), round(ymin), round(ymax))
                    if behaelter.empty:
                        display_str = "{}(EMPTY)".format(display_str)
                    else:
                        display_str = "{}(filled, {})".format(
                            display_str, behaelter.werk_stueck.color
                        )
                ################
                if category_index[classes[i]]["name"] == "WerkStueck":
                    pos = hochregallager.get_werkstueck_pos_by_werkstueck_box(
                        image, box)
                    if pos is None:
                        display_str2 = "POS: N/A".format(row, column)
                    else:
                        row, column = pos
                        display_str2 = "POS: {}x{}".format(row, column)
                    ymin, xmin, ymax, xmax = get_box_coord_relative_to_grid_coord(
                        image, box, hochregallager)
                    display_str2 = "{}, Left:{} Right:{} Top:{} Bottom:{}".format(
                        display_str2, round(xmin), round(xmax), round(ymin), round(ymax))

                    wkstk_color = detect_color_in_bounding_box(
                        image, box, use_normalized_coordinates
                    )
                    display_str = "{}(Color:{})".format(
                        display_str, wkstk_color)
                ################
            if not skip_scores:
                if not display_str:
                    display_str = "{}%".format(round(100 * scores[i]))
                else:
                    display_str = "{}: {}%".format(
                        display_str, round(100 * scores[i]))

            if display_str2 == "":
                display_str_list = display_str
            else:
                display_str_list = [str(display_str), str(display_str2)]

            str_list.append(display_str_list)
            # box_to_display_str_map[box].append(display_str_list)
            if agnostic_mode:
                box_to_color_map[box] = "DarkOrange"
            else:
                box_to_color_map[box] = STANDARD_COLORS[
                    classes[i] % len(STANDARD_COLORS)
                ]

    # Draw all boxes onto image.
    tmp_i = 0
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        # viz_utils.draw_bounding_box_on_image_array(
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        draw_bounding_box_on_image_tmp(
            image_pil,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=2,
            #thickness=0 if skip_boxes else line_thickness,
            # display_str_list=box_to_display_str_map[box],
            display_str_list=str_list[tmp_i],
            use_normalized_coordinates=use_normalized_coordinates,
        )
        np.copyto(image, np.array(image_pil))
        tmp_i += 1

    return image

#############################################################################################


def draw_bounding_box_on_image_tmp(image,
                                   ymin,
                                   xmin,
                                   ymax,
                                   xmax,
                                   color='red',
                                   thickness=4,
                                   display_str_list=(),
                                   use_normalized_coordinates=True):
    """
    Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=thickness,
                  fill=color)
    try:

        ################ FONT ###################
        font_size = 24
        font = ImageFont.truetype('arial.ttf', font_size)

        ################ FONT ###################

    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin
#############################################################################################


def visualize_missing_behaelter_timer(image, hochregallager):
    # prepare vars needed to determine grid cell coordinates
    ymin, xmin, _, _ = hochregallager.coordinates
    grid_width_in_px, grid_height_in_px = hochregallager.width_in_px, hochregallager.height_in_px
    percent = 1/3
    grid_cell_width = grid_width_in_px*percent
    grid_cell_height = grid_height_in_px*percent

    # putText only needs left and bottom coord
    bottom = ymin
    # grid shape: 3x3
    for row in range(3):
        bottom = bottom + (grid_cell_height)
        left = (xmin - grid_cell_width)
        for column in range(3):
            left = left + (grid_cell_width)

            if hochregallager.grid_cell_timer_arr[row][column] != 0:
                grid_cell_timer_val = get_grid_cell_timer_value(row, column)

                # visualize timer in image
                font = cv2.FONT_HERSHEY_SIMPLEX
                # bottom left corner (x,y)
                org = (int(left), int(bottom*0.95))
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(image, 'missing: {}s'.format(round(grid_cell_timer_val, 2)), org, font,
                                   fontScale, color, thickness)


# # Detect in Real-Time
category_index = label_map_util.create_category_index_from_labelmap(
    ANNOTATION_PATH + "/label_map.pbtxt"
)


# cap.release()


# Setup capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# In[71]:


# ARUCO
#aruco_params = cv2.aruco.DetectorParameters_create()
#aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
#####
#aruco_coord, _, _ = cv2.aruco.detectMarkers(image_np_with_detections, aruco_dict, parameters=aruco_params)
#cv2.polylines(image_np_with_detections, np.int64(aruco_coord), True, (0, 255, 100), 4)
#####


# In[72]:


#
hochregallager = Hochregallager()
#

# img = "IMG_20220616_095327"
# frame = cv2.imread("Tensorflow/workspace/images/train/{}.jpg".format(img))

while True:
    ret, frame = cap.read()
    image_np = np.array(frame)

# needed for while loop
    # reset behaelter_obj_list (avoid appending to list of previous frame)
    hochregallager.clear_behaelter_list()


# ret, frame = cap.read()
# image_np = np.array(frame)


    ####################  get detections  ##############################################################
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections

    # detection_classes should be ints.
    detections["detection_classes"] = detections["detection_classes"].astype(
        np.int64)


    image_np_with_detections = image_np.copy()
    ####################  get detections  ##############################################################


    # min score for bounding boxes
    min_score_threshold = 0.6
    label_id_offset = 1


    # filter detections by min score and classes
    boxes, classes, scores = filter_detections_by_score(
        detections["detection_boxes"],
        detections["detection_classes"] + label_id_offset,
        detections["detection_scores"],
        min_score_threshold
    )
    (
        filtered_WerkStueck_detections,
        filtered_Behaelter_detections,
    ) = filter_detections_by_class(boxes, classes, scores, category_index)


    # REMOVE
    #tmp_boxes, _, _ = filtered_Behaelter_detections
    # for i in range(tmp_boxes.shape[0]):
    #    im_height = image_np_with_detections.shape[0]
    #    im_width = image_np_with_detections.shape[1]
    #    top, left, bottom, right = get_box_coordinates_from_normalized_coordinates(tmp_boxes[i], im_height, im_width)
    #    start_point = (int(left), int(top))
    #    # Ending coordinate, here (220, 220)
    #    # represents the bottom right corner of rectangle
    #    end_point = (int(right), int(bottom))
    #    # Blue color in BGR
    #    color = (255, 255, 0)
    #    # Line thickness of 2 px
    #    thickness = 4
    #    # Using cv2.rectangle() method
    #    # Draw a rectangle with blue line borders of thickness of 2 px
    #    cv2.rectangle(image_np_with_detections, start_point, end_point, color, thickness)
    #    #cv2.polylines(image_np_with_detections, (top, left, bottom, right), True, (0, 0, 0), 4)
    #    cv2.imshow("object ", cv2.resize(image_np_with_detections, (1500, 1200)))
    #    cv2.waitKey(0)


    ############ Behaelter (and indirect Wkstk) initilaized ####################################################################
    # filtered_x_boxes, filtered_x_classes, filtered_x_scores = filtered_x_detections
    behaelter_boxes, _, behaelter_scores = filtered_Behaelter_detections


    for i in range(behaelter_boxes.shape[0]):
        box = tuple(behaelter_boxes[i].tolist())

        # REMOVE
        #im_height = image_np_with_detections.shape[0]
        #im_width = image_np_with_detections.shape[1]
        #top, left, bottom, right = get_box_coordinates_from_normalized_coordinates(box, im_height, im_width)
        #start_point = (int(left), int(top))
        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        #end_point = (int(right), int(bottom))
        # Blue color in BGR
        #color = (255, 255, 0)
        # Line thickness of 2 px
        #thickness = 4
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        #cv2.rectangle(image_np_with_detections, start_point, end_point, color, thickness)
        #cv2.polylines(image_np_with_detections, (top, left, bottom, right), True, (0, 0, 0), 4)
        #cv2.imshow("object ", cv2.resize(image_np_with_detections, (1500, 1200)))
        # cv2.waitKey(0)

        # Behaelter(..) also creates wkstk obj instance if its contained within
        behaelter = Behaelter(image_np_with_detections, box,
                              behaelter_scores[i], filtered_WerkStueck_detections)
        hochregallager.add_behaelter(behaelter)


    # handle grid initialize & assign
    if len(hochregallager.behaelter_obj_list) >= 1:
        if not hochregallager.grid_successfully_initialized:
            hochregallager.initialize_grid_coordinates()

        assign_grid_positions(hochregallager)

        print(len(hochregallager.behaelter_obj_list))
        print(hochregallager.behaelter_arr)

    ############ Behaelter (and indirect Wkstk) initilaized ####################################################################


    ##############  visualize  ################################################
    visualize_boxes_and_labels_for_behaelter_and_werkstueck(
        image_np_with_detections,
        boxes,
        classes,
        scores,
        category_index,
        visualize_werkstueck=False,
        behaelter_detections=filtered_Behaelter_detections,
        hochregallager=hochregallager,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=min_score_threshold,
        agnostic_mode=False,
    )


    cv2.imshow("object detection", cv2.resize(
        image_np_with_detections, (1500, 1200)))
    ##############  visualize  ################################################


# cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
cv2.destroyAllWindows()
# hochregallager.behaelter_arr


# cap = cv2.VideoCapture(0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# while True:
#     ret, frame = cap.read()
#     image_np = np.array(frame)
#
#     # ret, frame = cap.read()
#     image_np = np.array(frame)
#
#     #ymin, xmin, ymax, xmax = 100, 100, 1000, 1000
#
#     #str_list = ["A - 12", "B - 13"]
#
#     #image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
#     #draw_bounding_box_on_image_tmp(
#     #    image_pil,
#     #    ymin,
#     #    xmin,
#     #    ymax,
#     #    xmax,
#     #    color=color,
#     #    thickness=2,
#     #    #thickness=0 if skip_boxes else line_thickness,
#     #    display_str_list= str_list,
#     #    use_normalized_coordinates=False,
#     #)
#     #np.copyto(image_np, np.array(image_pil))
#
#
#     cv2.imshow("object detection", cv2.resize(image_np, (1280, 720)))
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap.release()
#         break
# cv2.destroyAllWindows()
