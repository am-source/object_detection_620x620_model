import cv2
import numpy as np
import coordinates as coord


# COLOR DETECT - get color for single bounding box


# COLOR THRESHOLDS
# 110 lower bounds to make sure white(/black) isn't mistakenly recognized
lower_sat = 60
lower_val = 25
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
        (top, left, bottom, right) = coord.get_box_coordinates_from_normalized_coordinates(
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
    # object_color = "UNDECIDED"
    # for mask in masks:
    #     if np.count_nonzero(mask[0]) > (mask[0].size * 0.5):
    #         object_color = mask[1]
    object_color = "UNDECIDED"
    max_mask_area = 0
    for mask in masks:
        if np.count_nonzero(mask[0]) > max_mask_area:
            max_mask_area = np.count_nonzero(mask[0])
            object_color = mask[1]

    return object_color
