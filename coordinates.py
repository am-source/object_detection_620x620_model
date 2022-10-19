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
    primary_box, secondary_boxes, secondary_boxes_scores=None, return_percent=False,
        percent_threshold=0.2
):
    p_ymin, p_xmin, p_ymax, p_xmax = primary_box

    intersect_elements = []

    # loop over all secondary bounding boxes and find the ones with intersecting areas
    for i in range(secondary_boxes.shape[0]):
        (
            s_ymin,
            s_xmin,
            s_ymax,
            s_xmax,
        ) = secondary_boxes[i]

        intersect_width = one_dim_intersect(p_xmin, p_xmax, s_xmin, s_xmax)
        intersect_height = one_dim_intersect(p_ymin, p_ymax, s_ymin, s_ymax)
        intersect_area = intersect_width * intersect_height

        # overlap percentage is based on the surface of SECONDARY box
        secondary_box_area = float((s_xmax - s_xmin) * (s_ymax - s_ymin))
        intersect_percent = intersect_area / secondary_box_area

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

# box coordinates have relative values (0-1) after detection, need to be changed to actual/absolute pixel values
def get_box_coordinates_from_normalized_coordinates(
    norm_box, image
):
    im_height = image.shape[0]
    im_width = image.shape[1]

    ymin, xmin, ymax, xmax = (
        norm_box[0] * im_height,
        norm_box[1] * im_width,
        norm_box[2] * im_height,
        norm_box[3] * im_width,
    )
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
        hochregallager.grid_successfully_initialized = False
        return None
    # calc grid outline using behaelter coords
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
    # corners is a list of points, each point has the form (x_val,y_val)
    # initialize values with the first point
    ymin, xmin, ymax, xmax = corners[0][1], corners[0][0], corners[0][1], corners[0][0]

    # find  ymin,xmin,ymax,xmax of the aruco marker
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
    # change contrast to make aruco detection easier, reduces influence of lighting
    current_frame = modify_contrast(current_frame)

    # detectMarkers returns multiple(!) objects for each return val
    aruco_bboxes, aruco_ids, _ = cv2.aruco.detectMarkers(current_frame, aruco_dict, parameters=aruco_params)

    # left marker probably wasn't detected, try with hard contrast/value changes
    if len(aruco_bboxes) == 1:
        contrast_mask_lower = np.array([0, 0, 230], np.uint8)
        contrast_mask_upper = np.array([179, 27, 255], np.uint8)

        # HSV based frame
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # create mask of the frame
        contrast_mask = cv2.inRange(hsv_frame, contrast_mask_lower, contrast_mask_upper)
        aruco_bboxes_2, aruco_ids_2, _ = cv2.aruco.detectMarkers(contrast_mask, aruco_dict, parameters=aruco_params)
        for i in range(len(aruco_bboxes_2)):
            if aruco_ids_2[i] != aruco_ids[0]:
                aruco_bboxes = [aruco_bboxes[0], aruco_bboxes_2[i]]
                break

    return aruco_bboxes, aruco_ids


def modify_contrast(image, contrast=60):
    # contrast of 60 seems to work best
    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha = f
    beta = 0
    gamma = 127*(1-f)

    image = cv2.addWeighted(image, alpha, image, beta, gamma)
    return image


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
        # get a list of temporarily assigned positions (not yet assigned in HRL)
        tmp_behaelter_arr = get_tmp_grid_positions(hochregallager)

        # assign positions (based on tmp_behaelter_list) and handle missing timer
        for i in range(3):
            for j in range(3):
                behaelter_obj = tmp_behaelter_arr[i][j]
                if behaelter_obj is not None:
                    # grid cell is filled and was empty
                    if hochregallager.behaelter_arr[i][j] is None:
                        hochregallager.assign_grid_pos(
                            behaelter_obj, row=i, column=j)
                        # covers the case where this is the first time this position is filled after launching the
                        # program. Since no timer was started beforehand, there is no running timer to stop
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
    tmp_list = hochregallager.behaelter_obj_list.copy()
    tmp_list = [elem.bounding_box for elem in tmp_list]
    behaelter_np_arr = np.array(tmp_list)

    # create 3x3 array to temporarily act as Hochregallager.behaelter_arr, if the grid doesnt fulfill set conditions
    # then no behaelter should actually be assigned to the real Hochregallager.behaelter_arr
    tmp_behaelter_arr = [[None for x in range(3)] for y in range(3)]

    # this percent is only used for cell width, since the webcam angle only slightly distorts the cell width
    # adjustments are not needed (unlike cell heights)
    percent = 1 / 3
    grid_cell_width = grid_width_in_px * percent

    # loop over all POSs intmp_behaelter_arr and change status accordingly
    for i in range(3):
        left, right = (xmin - grid_cell_width), xmin
        for j in range(3):
            # top and bottom of cells are not all equal in height, need to be adjusted to account for webcam angle
            top, bottom = get_grid_cell_top_and_bottom(hochregallager, row=i, column=j)
            left = left + (grid_cell_width)
            right = right + (grid_cell_width)

            grid_cell_bounding_box = (top, left, bottom, right)
            intersect_elements = bounding_box_intersect(grid_cell_bounding_box, behaelter_np_arr, return_percent=True)

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
                bool_mask = []
                for k in range(behaelter_np_arr.shape[0]):
                    condition = False if tuple(behaelter_np_arr[k]) == actual_intersect_elem[0] else True
                    bool_mask.append(condition)
                behaelter_np_arr = behaelter_np_arr[bool_mask]

                for elem in hochregallager.behaelter_obj_list:
                    if elem.bounding_box == actual_intersect_elem[0]:
                        behaelter_obj = elem

                # save Behaelter obj in temporary behaelter_arr
                tmp_behaelter_arr[i][j] = behaelter_obj
    return tmp_behaelter_arr


def get_grid_cell_top_and_bottom(hochregallager, row, column):
    """
    percent value shows the y axis distribution of the grid cell height
                        _____________________
                        || 18% | 30% | 50% ||
                        ---------------------
                        || 32% | 30% | 32% ||
                        ---------------------
                        || 50% | 40% | 18% ||
                        ---------------------
                        ||                 ||
                        ||                 ||

    """
    # if column == 0:
    #     if row == 0:
    #         percent_top = 0
    #         percent_bottom = 1/3
    #     elif row == 1:
    #         percent_top = 1/3
    #         percent_bottom = 2/3
    #     else:
    #         percent_top = 2/3
    #         percent_bottom = 1
    #
    # elif column == 1:
    #     if row == 0:
    #         percent_top = 0
    #         percent_bottom = 0.3
    #     elif row == 1:
    #         percent_top = 0.3
    #         percent_bottom = 0.6
    #     else:
    #         percent_top = 0.6
    #         percent_bottom = 1
    #
    # else:
    #     if row == 0:
    #         percent_top = 0
    #         percent_bottom = 0.3
    #     elif row == 1:
    #         percent_top = 0.3
    #         percent_bottom = 0.6
    #     else:
    #         percent_top = 0.6
    #         percent_bottom = 1
    if column == 0:
        if row == 0:
            percent_top = 0
            percent_bottom = 0.18
        elif row == 1:
            percent_top = 0.18
            percent_bottom = 0.5
        else:
            percent_top = 0.5
            percent_bottom = 1

    elif column == 1:
        if row == 0:
            percent_top = 0
            percent_bottom = 0.3
        elif row == 1:
            percent_top = 0.3
            percent_bottom = 0.6
        else:
            percent_top = 0.6
            percent_bottom = 1

    else:
        if row == 0:
            percent_top = 0
            percent_bottom = 0.5
        elif row == 1:
            percent_top = 0.5
            percent_bottom = 0.82
        else:
            percent_top = 0.82
            percent_bottom = 1

    grid_height = hochregallager.height_in_px
    grid_top = hochregallager.coordinates[0]

    cell_top = grid_top + grid_height * percent_top
    cell_bottom = grid_top + grid_height * percent_bottom
    return cell_top, cell_bottom


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


# used for cards on left sid of web UI, behaelter coord
def get_box_coord_relative_to_grid_coord(box, hochregallager):
    h_ymin, h_xmin, _, _ = hochregallager.coordinates
    b_ymin, b_xmin, b_ymax, b_xmax = box
    relative_ymin = b_ymin - h_ymin
    relative_xmin = b_xmin - h_xmin
    relative_ymax = b_ymax - h_ymin
    relative_xmax = b_xmax - h_xmin
    return (relative_ymin, relative_xmin, relative_ymax, relative_xmax)
