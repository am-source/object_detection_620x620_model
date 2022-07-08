import threading
import cv2
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
import os
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
import numpy as np
from flask import Flask, render_template, Response, jsonify
import random

app = Flask(__name__)

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
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-21')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# # Detect in Real-Time
category_index = label_map_util.create_category_index_from_labelmap(
    ANNOTATION_PATH + "/label_map.pbtxt"
)


def run_video():
    # grab global references to the video stream, output frame, and
    # lock variables
    global cap, outputFrame, lock
    outputFrame= None
    total = 0
    # loop over frames from the video stream
    while True:
        cap = cv2.VideoCapture(0)
        while (cap.isOpened()):
            ret, img = cap.read()
            if ret == True:
                total += 1
                with lock:
                    outputFrame = img.copy()

def generate():
    # global outputFrame, lock, prev_response_time
    while True:
        with lock:
            if outputFrame is None:
                continue

            ###############################################
            image_np = np.array(outputFrame)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=.5,
                agnostic_mode=False)

            # frame = cv2.resize(image_np_with_detections, (640, 480))
            # ret, buffer = cv2.imencode('.jpg', frame)
            # frame = buffer.tobytes()
            #
            # yield (
            #         b'--frame\r\n'
            #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            # )
            ###############################################
            (flag, encodedImage) = cv2.imencode(".jpg", image_np_with_detections)
            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route('/video')
def video():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")




@app.route('/_stuff', methods=['GET'])
def stuff():
    pos_dict = {
        "pos_0x0": random.randint(0, 9),
        "pos_0x1": random.randint(0, 9),
        "pos_0x2": random.randint(0, 9),
        "pos_1x0": random.randint(0, 9),
        "pos_1x1": random.randint(0, 9),
        "pos_1x2": random.randint(0, 9),
        "pos_2x0": random.randint(0, 9),
        "pos_2x1": random.randint(0, 9),
        "pos_2x2": random.randint(0, 9)
    }
    return jsonify(pos_dict=pos_dict)

@app.route('/')
def index():
    return render_template('index.html')


# check to see if this is the main thread of execution
lock = threading.Lock()
if __name__ == '__main__':
    thread = threading.Thread(target=run_video)
    thread.daemon = True
    thread.start()
    # start the flask app
    app.run(threaded=True)
