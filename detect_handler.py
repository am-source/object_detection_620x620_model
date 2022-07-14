from camera import MyCamera
from hoch_regal_lager import Hochregallager
import tensorflow as tf
from object_detection.utils import config_util
import os
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
import numpy as np
import visualize
from behaelter import Behaelter
import coordinates as coord


WORKSPACE_PATH = "Tensorflow/workspace"
ANNOTATION_PATH = WORKSPACE_PATH + "/annotations"
MODEL_PATH = WORKSPACE_PATH + "/models"
CHECKPOINT_PATH = MODEL_PATH + "/my_ssd_mobnet/"
CUSTOM_MODEL_NAME = "my_ssd_mobnet"
CONFIG_PATH = MODEL_PATH + "/" + CUSTOM_MODEL_NAME + "/pipeline.config"


## TF setup can't be called more than once, otherwise error would occur
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(
    model_config=configs["model"], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, "ckpt-21")).expect_partial()


def handle_detection(camera, hochregallager):
    frame = camera.frame
    image_np = np.array(frame)

    # needed for while loop
    # reset behaelter_obj_list (avoid appending to list of previous frame)
    hochregallager.clear_behaelter_list()
    if hochregallager.image is None:
        hochregallager.set_image(image_np)

    ####################  get detections  ##############################################################
    category_index = get_category_index()
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor, detection_model)

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
    boxes, classes, scores = visualize.filter_detections_by_score(
        detections["detection_boxes"],
        detections["detection_classes"] + label_id_offset,
        detections["detection_scores"],
        min_score_threshold
    )
    (
        filtered_WerkStueck_detections,
        filtered_Behaelter_detections,
    ) = visualize.filter_detections_by_class(boxes, classes, scores, category_index)

    # ############ Behaelter (and indirect Wkstk) initilaized #####################################
    initialize_and_handle_objects(image_np_with_detections, hochregallager, filtered_Behaelter_detections, filtered_WerkStueck_detections)
    # ############ Behaelter (and indirect Wkstk) initilaized #####################################

    ##############  visualize  ################################################
    # line_thickness concerns box outline, depending on webcam resolution (and possibly distance of objects) needs to be modified
    # skip_boxes sets line_thickness to 0
    # skip_werkstueck: whether to skip visualizing WerkStueck class
    # skip_missing_timer: whether to skip visualizing timer in sec at grid cell position
    # text_font_size: depending on webcam resolution (and possibly distance of objects) needs to be modified
    visualize.visualize_boxes_and_labels_for_behaelter_and_werkstueck(
        image_np_with_detections,
        boxes,
        classes,
        scores,
        category_index,
        behaelter_detections=filtered_Behaelter_detections,
        hochregallager=hochregallager,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        agnostic_mode=True,
        line_thickness=2,
        skip_werkstueck=True,
        skip_missing_timer=False,
        skip_boxes=False,
        skip_scores=True,
        skip_labels=False,
        text_font_size=16,
    )

    return image_np_with_detections

    # cv2.imshow("object detection", cv2.resize(
    #     image_np_with_detections, (1500, 1200)))

    # image_np_with_detections = cv2.resize(image_np_with_detections, (1280, 720))
    ##############  visualize  ################################################


@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def get_category_index():
    category_index = label_map_util.create_category_index_from_labelmap(
        ANNOTATION_PATH + "/label_map.pbtxt"
    )
    return category_index


def initialize_and_handle_objects(image_np_with_detections, hochregallager, filtered_Behaelter_detections, filtered_WerkStueck_detections):
    # filtered_x_boxes, filtered_x_classes, filtered_x_scores = filtered_x_detections
    behaelter_boxes, _, behaelter_scores = filtered_Behaelter_detections
    # loop over all (filtered) boxes
    for i in range(behaelter_boxes.shape[0]):
        box = tuple(behaelter_boxes[i].tolist())

        # Behaelter(..) also creates wkstk obj instance if its contained within
        behaelter = Behaelter(image_np_with_detections, box,
                              behaelter_scores[i], filtered_WerkStueck_detections)
        hochregallager.add_behaelter(behaelter)

    # handle grid initialize & assign
    # objects need to be present to process
    if len(hochregallager.behaelter_obj_list) >= 1:
        # only assign grid pos if 9 Behaelter were detected, otherwise the positions would be wrong
        if hochregallager.grid_successfully_initialized:
            coord.assign_grid_positions(hochregallager)
        # if less than 9 Behaelter were detected in the last iteration, try to initialize the grid again
        else:
            hochregallager.initialize_grid_coordinates()
