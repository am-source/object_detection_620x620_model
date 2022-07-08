#!/usr/bin/env python
# coding: utf-8
import time
import cv2
import tensorflow as tf
from object_detection.utils import config_util
import os
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
import numpy as np
from flask import Flask, render_template, Response, jsonify
from hoch_regal_lager import Hochregallager

# # Setup

# REMOVE
# SCRIPTS_PATH = "Tensorflow/scripts"
# APIMODEL_PATH = "Tensorflow/models"
# IMAGE_PATH = WORKSPACE_PATH + "/images"
# PRETRAINED_MODEL_PATH = WORKSPACE_PATH + "/pre-trained-models"
WORKSPACE_PATH = "Tensorflow/workspace"
ANNOTATION_PATH = WORKSPACE_PATH + "/annotations"
MODEL_PATH = WORKSPACE_PATH + "/models"
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

category_index = label_map_util.create_category_index_from_labelmap(
    ANNOTATION_PATH + "/label_map.pbtxt"
)


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections






########## FLASK ROUTES ############################################################################
def generate_frames():
    cap = cv2.VideoCapture(0)
    print("ENTERED generate_frames!")
    while True:
        ret, frame = cap.read()
        print("IN while True BLOCK!")
        if not ret:
            continue
        print("IN while True BLOCK! - IF ret WAS TRUE")
        image_np = np.array(frame)

        # needed for while loop
        # reset behaelter_obj_list (avoid appending to list of previous frame)
        hochregallager.clear_behaelter_list()
        if hochregallager.image is None:
            hochregallager.set_image(image_np)

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

        ############ Behaelter (and indirect Wkstk) initilaized ####################################################################
        # filtered_x_boxes, filtered_x_classes, filtered_x_scores = filtered_x_detections
        behaelter_boxes, _, behaelter_scores = filtered_Behaelter_detections


        for i in range(behaelter_boxes.shape[0]):
            box = tuple(behaelter_boxes[i].tolist())

            # Behaelter(..) also creates wkstk obj instance if its contained within
            behaelter = Behaelter(image_np_with_detections, box,
                                  behaelter_scores[i], filtered_WerkStueck_detections)
            hochregallager.add_behaelter(behaelter)


        # handle grid initialize & assign
        if len(hochregallager.behaelter_obj_list) >= 1:
            if not hochregallager.grid_successfully_initialized:
                hochregallager.initialize_grid_coordinates()

            assign_grid_positions(image_np_with_detections, hochregallager)

            # print(len(hochregallager.behaelter_obj_list))
            # print(hochregallager.behaelter_arr)

        ############ Behaelter (and indirect Wkstk) initilaized ####################################################################


        ##############  visualize  ################################################
        # visualize_boxes_and_labels_for_behaelter_and_werkstueck(
        #     image_np_with_detections,
        #     boxes,
        #     classes,
        #     scores,
        #     category_index,
        #     visualize_werkstueck=False,
        #     behaelter_detections=filtered_Behaelter_detections,
        #     hochregallager=hochregallager,
        #     use_normalized_coordinates=True,
        #     max_boxes_to_draw=20,
        #     min_score_thresh=min_score_threshold,
        #     agnostic_mode=False,
        # )


        # cv2.imshow("object detection", cv2.resize(
        #     image_np_with_detections, (1500, 1200)))

        # image_np_with_detections = cv2.resize(image_np_with_detections, (1280, 720))
        ret, buffer = cv2.imencode('.jpg', image_np_with_detections)
        image_np_with_detections = buffer.tobytes()

        yield(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image_np_with_detections + b'\r\n'
        )
        time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/_stuff', methods=['GET'])
def stuff():
    # coords
    # score
    # filled
    # wkstk score(?)
    # color
    # missing

    behaelter_dict = {}
    # behaelter (dicts) are numbered 1-9
    dict_key_var = 1
    for row in range(3):
        for column in range(3):
            if hochregallager.behaelter_arr[row][column] is None:
                coords = "N/A"
                score = "N/A"
                filled = "N/A"
                wkstk_score = "N/A"
                color = "N/A"
                # in case the behaelter hasn't been detected yet (for the first time)
                if hochregallager.grid_cell_timer_arr[row][column] == 0 and not hochregallager.grid_successfully_initialized:
                    missing = "N/A"
                else:
                    missing = str(get_grid_cell_timer_value(row,column))

            else:
                behaelter = hochregallager.behaelter_arr[row][column]
                ymin, xmin, ymax, xmax = get_box_coord_relative_to_grid_coord(hochregallager.image, behaelter.bounding_box, hochregallager)
                coords = str((round(ymin, 2), round(xmin, 2), round(ymax, 2), round(xmax, 2)))
                score = str(behaelter.score)
                filled = not behaelter.empty
                if filled:
                    wkstk_score = str(behaelter.werk_stueck.score)
                    color = str(behaelter.werk_stueck.color)
                else:
                    wkstk_score = "N/A"
                    color = "N/A"
                missing = "N/A"

            behaelter_dict[dict_key_var] = {
                # coords = ymin, xmin, ymax, xmax or "N/A"
                "coords": coords,
                "score": score,
                "filled": filled,
                "wkstk_score": wkstk_score,
                "color": color,
                "missing": missing
            }

            dict_key_var += 1
    return jsonify(pos_dict=behaelter_dict)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
########## FLASK ROUTES ############################################################################


app = Flask(__name__)

if __name__ == "__main__":
    hochregallager = Hochregallager()

    # app.run(debug=True)
    app.run()
