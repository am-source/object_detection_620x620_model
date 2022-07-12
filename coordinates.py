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
# should only be called when hochregallager.behaelter_obj_list has AT LEAST one behaelter
def get_approx_hochregallager_grid_coordinates(hochregallager):
    behaelter_list = hochregallager.behaelter_obj_list

    ymin_test, xmin_test, ymax_test, xmax_test = behaelter_list[0].bounding_box[0], behaelter_list[
        0].bounding_box[1], behaelter_list[0].bounding_box[2], behaelter_list[0].bounding_box[3]

    for behaelter in behaelter_list:
        new_ymin, new_xmin, new_ymax, new_xmax = behaelter.bounding_box
        if ymin_test > new_ymin:
            ymin_test = new_ymin
        if xmin_test > new_xmin:
            xmin_test = new_xmin
        if ymax_test < new_ymax:
            ymax_test = new_ymax
        if xmax_test < new_xmax:
            xmax_test = new_xmax
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
                # print('No behaelter found, POS:{}x{}'.format(i, j))
                # grid cell is empty and was empty
                if hochregallager.behaelter_arr[i][j] is None:
                    pass
                # grid cell is empty and was filled before
                else:
                    hochregallager.remove_behaelter(row=i, column=j)
                    current_time = time.time()
                    hochregallager.start_grid_cell_timer(
                        row=i, column=j, current_time=current_time)


def get_box_coord_relative_to_grid_coord(box, hochregallager):
    h_ymin, h_xmin, _, _ = hochregallager.coordinates
    b_ymin, b_xmin, b_ymax, b_xmax = box
    relative_ymin = b_ymin - h_ymin
    relative_xmin = b_xmin - h_xmin
    relative_ymax = b_ymax - h_ymin
    relative_xmax = b_xmax - h_xmin
    return (relative_ymin, relative_xmin, relative_ymax, relative_xmax)
