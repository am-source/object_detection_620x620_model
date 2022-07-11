import cv2
import collections
from object_detection.utils import visualization_utils as viz_utils
import six
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.Image as Image
import numpy as np
import coordinates as coord
from hoch_regal_lager import Hochregallager
import color_detector


# # Visualization
# @override visualization_utils.visualize_boxes_and_labels_on_image_array
# Modified version - removed all unnecessary parts
STANDARD_COLORS = viz_utils.STANDARD_COLORS


def visualize_boxes_and_labels_for_behaelter_and_werkstueck(
    image,
    boxes,
    classes,
    scores,
    category_index,
    behaelter_detections,
    hochregallager,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color="black",
    skip_werkstueck=True,
    skip_missing_timer = True,
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    text_font_size = 16,
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
    skip_werkstueck: whether to skip visualizing WerkStueck class
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
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    if not skip_missing_timer:
        visualize_missing_behaelter_timer(image_pil, hochregallager, text_font_size)
        np.copyto(image, np.array(image_pil))

    # by default only visualize 'Behaelter'
    if skip_werkstueck:
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
                    # print("FIRST pos: {}".format(pos))
                    behaelter = hochregallager.get_behaelter_obj_by_behaelter_box(
                        image, box)
                    if pos is None:
                        display_str2 = "POS: N/A"
                    else:
                        row, column = pos
                        display_str2 = "POS: {}x{}".format(row, column)

                    ymin, xmin, ymax, xmax = coord.get_box_coord_relative_to_grid_coord(
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
                        display_str2 = "POS: N/A"
                    else:
                        row, column = pos
                        display_str2 = "POS: {}x{}".format(row, column)
                    ymin, xmin, ymax, xmax = coord.get_box_coord_relative_to_grid_coord(
                        image, box, hochregallager)
                    display_str2 = "{}, Left:{} Right:{} Top:{} Bottom:{}".format(
                        display_str2, round(xmin), round(xmax), round(ymin), round(ymax))

                    wkstk_color = color_detector.detect_color_in_bounding_box(
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
            thickness=0 if skip_boxes else line_thickness,
            # display_str_list=box_to_display_str_map[box],
            display_str_list=str_list[tmp_i],
            use_normalized_coordinates=use_normalized_coordinates,
            bounding_box_font_size=text_font_size,
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
                                   use_normalized_coordinates=True,
                                   bounding_box_font_size=24):
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
        font_size = bounding_box_font_size
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


def visualize_missing_behaelter_timer(image, hochregallager, text_font_size):
    draw = ImageDraw.Draw(image)
    # prepare vars needed to determine grid cell coordinates
    ymin, xmin, _, _ = hochregallager.coordinates
    grid_width_in_px, grid_height_in_px = hochregallager.width_in_px, hochregallager.height_in_px
    percent = 1/3
    grid_cell_width = grid_width_in_px*percent
    grid_cell_height = grid_height_in_px*percent

    # put text in centre of the grid cell
    top = ymin - (grid_cell_height*0.5)
    # grid shape: 3x3
    for row in range(3):
        top = top + (grid_cell_height)
        left = (xmin - grid_cell_width)
        for column in range(3):
            left = left + (grid_cell_width)

            if hochregallager.grid_cell_timer_arr[row][column] != 0:
                grid_cell_timer_val = hochregallager.get_grid_cell_timer_value(row, column)

                try:
                    ################ FONT ###################
                    font_size = text_font_size
                    font = ImageFont.truetype('arial.ttf', font_size)
                    ################ FONT ###################

                except IOError:
                    font = ImageFont.load_default()
                display_str = 'missing:\n{}s'.format(round(grid_cell_timer_val, 2))
                text_width, text_height = font.getsize(display_str)
                text_bottom = top
                margin = np.ceil(0.05 * text_height)
                left = left + (grid_cell_width*0.5)
                draw.text(
                    (left + margin, text_bottom - text_height - margin),
                    display_str,
                    # fill takes RGBA value, A stands for alpha - opacity value
                    fill=(0, 0, 255, 255),
                    font=font,
                    align='center')

# # Filter methods

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
