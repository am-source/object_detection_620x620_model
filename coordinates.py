import time
import cv2
import numpy as np

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


def bounding_box_intersect(
    primary_box, secondary_boxes, secondary_boxes_scores=None, im_height=None, im_width=None, needs_normalization=True, return_percent=False, percent_threshold=0.2
):
    p_ymin, p_xmin, p_ymax, p_xmax = primary_box
    #primary_box_area = float((p_xmax - p_xmin) * (p_ymax - p_ymin))
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

        secondary_box_area = float((s_xmax - s_xmin) * (s_ymax - s_ymin))
        intersect_percent = intersect_area / secondary_box_area
        # REMOVE
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


# get coordinates for lowest ymin xmin, highest ymax xmax to find the 4 corners
def get_approx_hochregallager_grid_coordinates(hochregallager):
    aruco_bboxes, aruco_ids = get_aruco_markers(hochregallager)

    # only possible if both markers are detected
    if len(aruco_bboxes) == 2:
        # since the outline of the Hochregallager is visible, we can assume that Behaelter can be assigned to grid cells
        hochregallager.grid_successfully_initialized = True

        # aruco_bboxes consists of a list length 2 (for every marker), the list for a marker consists of a list with
        # only 1 element, this element is a list of 4 points, each in form of (x_val,y_val)
        # get ymin, xmin, ymax, xmax for every marker
        b1_ymin, b1_xmin, b1_ymax, b1_xmax = handle_aruco_corner(aruco_bboxes[0][0])
        b2_ymin, b2_xmin, b2_ymax, b2_xmax = handle_aruco_corner(aruco_bboxes[1][0])

        # if b1 is the left marker, then take it's right(xmax) as xmin (since its left side would unnecessarily distort
        # the grid, take b1's ymin and the max values from the right marker
        if b1_xmin < b2_xmin:
            grid_xmin = b1_xmax
            grid_xmax = b2_xmin
            grid_ymin = b1_ymin
            grid_ymax = b2_ymin
        # b1 is the right marker
        else:
            grid_xmin = b2_xmax
            grid_xmax = b1_xmin
            grid_ymin = b2_ymin
            grid_ymax = b1_ymin
        return (grid_ymin, grid_xmin, grid_ymax, grid_xmax)
    else:
        return None
    # behaelter_list = hochregallager.behaelter_obj_list
    #
    # ymin_test, xmin_test, ymax_test, xmax_test = behaelter_list[0].bounding_box[0], behaelter_list[
    #     0].bounding_box[1], behaelter_list[0].bounding_box[2], behaelter_list[0].bounding_box[3]
    #
    # for behaelter in behaelter_list:
    #     new_ymin, new_xmin, new_ymax, new_xmax = behaelter.bounding_box
    #     if ymin_test > new_ymin:
    #         ymin_test = new_ymin
    #     if xmin_test > new_xmin:
    #         xmin_test = new_xmin
    #     if ymax_test < new_ymax:
    #         ymax_test = new_ymax
    #     if xmax_test < new_xmax:
    #         xmax_test = new_xmax
    # return (ymin_test, xmin_test, ymax_test, xmax_test)


def handle_aruco_corner(corners):
    # corners is a least of points, each point has the form (x_val,y_val)
    ymin, xmin, ymax, xmax = corners[0][1], corners[0][0], corners[0][1], corners[0][0]

    for point in corners:
        if point[0] < xmin:
            xmin = point[0]
        if point[0] > xmax:
            xmax = point[0]
        if point[1] < ymin:
            ymin = point[1]
        if point[1] > ymax:
            ymax = point[1]

    return ymin, xmin, ymax, xmax


def get_aruco_markers(hochregallager):
    # setup for marker detection
    aruco_params = cv2.aruco.DetectorParameters_create()
    # smallest dictionary, since only 2 markers are needed and it's easier to detect simpler patterns
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    current_frame = hochregallager.image.copy()
    # gray scale should help improve the marker detection
    gray_img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # detectMarkers returns multiple(!) objects for each return val
    aruco_bboxes, aruco_ids, _ = cv2.aruco.detectMarkers(gray_img, aruco_dict, parameters=aruco_params)
    return aruco_bboxes, aruco_ids


def get_approx_hochregallager_grid_width(hochregallager):
    _, xmin, _, xmax = hochregallager.coordinates
    return xmax-xmin


def get_approx_hochregallager_grid_height(hochregallager):
    ymin, _, ymax, _ = hochregallager.coordinates
    return ymax-ymin


def handle_grid_positions(hochregallager):
    # tmp_behaelter_arr = get_tmp_grid_positions(hochregallager)
    # grid_init_successful = check_grid_init_successful(tmp_behaelter_arr)

    # if grid_init_successful:
    #     hochregallager.grid_successfully_initialized = True
    if hochregallager.grid_successfully_initialized:
        tmp_behaelter_arr = get_tmp_grid_positions(hochregallager)
        for i in range(3):
            for j in range(3):
                behaelter_obj = tmp_behaelter_arr[i][j]
                if behaelter_obj is not None:
                    # grid cell is filled and was empty
                    if hochregallager.behaelter_arr[i][j] is None:
                        hochregallager.assign_grid_pos(
                            behaelter_obj, row=i, column=j)
                        # covers the case where this is the first iteration after launching the program. Since no timer was
                        # started beforehand, there is no running timer to stop
                        if hochregallager.grid_cell_timer_arr[i][j] == 0:
                            pass
                        else:
                            hochregallager.stop_grid_cell_timer(row=i, column=j)
                    # grid cell is filled and was filled before
                    else:
                        # update behaelter obj, possibly changing wkstk color and/or scores
                        hochregallager.assign_grid_pos(
                            behaelter_obj, row=i, column=j)

                # no behaelter found in grid cell
                else:
                    # grid cell is empty and was empty
                    if hochregallager.behaelter_arr[i][j] is None:
                        pass
                    # grid cell is empty and was filled before
                    else:
                        hochregallager.remove_behaelter(row=i, column=j)
                        current_time = time.time()
                        hochregallager.start_grid_cell_timer(
                            row=i, column=j, current_time=current_time)

    # modify boolean to represent that currently the coordinates of the grid cannot/shouldn't be accessed
    # else:
    #     hochregallager.grid_successfully_initialized = False


def get_tmp_grid_positions(hochregallager):
    ymin, xmin, _, _ = hochregallager.coordinates
    grid_width_in_px, grid_height_in_px = hochregallager.width_in_px, hochregallager.height_in_px

    # create copy of behaelter_obj_list, remove already assigned behaelter only from this copy
    tmp_behaelter_obj_list = hochregallager.behaelter_obj_list.copy()

    # create 3x3 array to temporarily act as Hochregallager.behaelter_arr, if the grid doesnt fulfill set conditions
    # then no behaelter should actually be assigned to the real Hochregallager.behaelter_arr
    tmp_behaelter_arr = [[None for x in range(3)] for y in range(3)]

    percent = 1 / 3
    grid_cell_width = grid_width_in_px * percent
    grid_cell_height = grid_height_in_px * percent

    top = ymin - grid_cell_height
    bottom = ymin

    # loop over all POSs intmp_behaelter_arr and change status accordingly
    for i in range(3):
        top = top + (grid_cell_height)
        bottom = bottom + (grid_cell_height)
        left, right = (xmin - grid_cell_width), xmin
        for j in range(3):
            left = left + (grid_cell_width)
            right = right + (grid_cell_width)

            # create a numpy 2d arr for bounding boxes of hochregallager.behaelter_obj_list (needed for bounding_box_intersect)
            tmp_list = []
            for obj in tmp_behaelter_obj_list:
                tmp_list.append(obj.bounding_box)
            behaelter_np_arr = np.array(tmp_list)

            grid_cell_bounding_box = (top, left, bottom, right)
            intersect_elements = bounding_box_intersect(grid_cell_bounding_box, behaelter_np_arr, needs_normalization=False, return_percent=True)

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

                # remove elem to avoid multiple assignment and avoid unnecessary iterations
                for obj in tmp_behaelter_obj_list:
                    if actual_intersect_elem[0] == obj.bounding_box:
                        tmp_behaelter_obj_list.remove(obj)

                # find corresponding behaelter object and assign to grid
                for elem in hochregallager.behaelter_obj_list:
                    if elem.bounding_box == actual_intersect_elem[0]:
                        behaelter_obj = elem

                # save Behaelter obj in temporary behaelter_arr
                tmp_behaelter_arr[i][j] = behaelter_obj
    return tmp_behaelter_arr


# def check_grid_init_successful(tmp_behaelter_arr):
#     # row 0, row 2, column 0, column 2 each need at least one Behaelter to be able to pinpoint the outline of the grid
#     # row 0 is responsible for ymin value , column 0 is responsible for xmin value
#     # row 2 is responsible for ymax value , column 2 is responsible for xmax value
#     row_0 = row_2 = column_0 = column_2 = False
#     for i in range(3):
#         if tmp_behaelter_arr[0][i] is not None:
#             row_0 = True
#         if tmp_behaelter_arr[2][i] is not None:
#             row_2 = True
#         if tmp_behaelter_arr[i][0] is not None:
#             column_0 = True
#         if tmp_behaelter_arr[i][2] is not None:
#             column_2 = True
#
#     result = row_0 and row_2 and column_0 and column_2
#     return result


def get_box_coord_relative_to_grid_coord(box, hochregallager):
    h_ymin, h_xmin, _, _ = hochregallager.coordinates
    b_ymin, b_xmin, b_ymax, b_xmax = box
    relative_ymin = b_ymin - h_ymin
    relative_xmin = b_xmin - h_xmin
    relative_ymax = b_ymax - h_ymin
    relative_xmax = b_xmax - h_xmin
    return (relative_ymin, relative_xmin, relative_ymax, relative_xmax)
