#!/usr/bin/env python
# coding: utf-8
import cv2
import detect_handler as detect
from flask import Flask, render_template, Response, jsonify, request
from hoch_regal_lager import Hochregallager
from camera import MyCamera
import coordinates


app = Flask(__name__)


########## FLASK ROUTES ############################################################################
@app.route('/handle_resolution', methods=['GET'])
def handle_resolution():
    fst_res = int(request.args['fst_res'])
    snd_res = int(request.args['snd_res'])
    if fst_res > 0 and snd_res > 0:
        global cam_resolution
        cam_resolution = (fst_res, snd_res)
    # need to send some response -> just reload
    return render_template('index.html')

def generate_frames():
    cam = MyCamera()
    # print("ENTERED generate_frames!")
    while True:
        # print("IN while True BLOCK!")
        if not cam.frame_available():
            continue
        # print("IN while True BLOCK! - IF ret WAS TRUE")

        image_np_with_detections = detect.handle_detection(camera=cam, hochregallager=hochregallager)
        # resize to match original resolution
        image_np_with_detections = cv2.resize(image_np_with_detections, cam_resolution)
        # encode for Response stream
        ret, buffer = cv2.imencode('.jpg', image_np_with_detections)
        image_np_with_detections = buffer.tobytes()

        yield(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image_np_with_detections + b'\r\n'
        )


@app.route('/card_update', methods=['GET'])
def card_update():
    # dict keys: coords, score, filled, wkstk_score, color, missing
    behaelter_dict = create_behaelter_dict()
    return jsonify(pos_dict=behaelter_dict)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
########## FLASK ROUTES ############################################################################


def create_behaelter_dict():
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
            else:
                behaelter = hochregallager.behaelter_arr[row][column]
                if hochregallager.grid_successfully_initialized:
                    ymin, xmin, _, _ = coordinates.get_box_coord_relative_to_grid_coord(
                        behaelter.bounding_box, hochregallager
                    )
                    coords = str((round(ymin, 2), round(xmin, 2)))
                # grid isn't accessible
                else:
                    coords = "N/A"
                score = str(behaelter.score)
                filled = not behaelter.empty
                if filled:
                    wkstk_score = str(behaelter.werk_stueck.score)
                    color = str(behaelter.werk_stueck.color)
                else:
                    wkstk_score = "N/A"
                    color = "N/A"

            if hochregallager.grid_cell_timer_arr[row][column] == 0:
                missing = str(0)
            else:
                missing = str(hochregallager.get_grid_cell_timer_value(row, column))

            missing_record_list = hochregallager.missing_time_record_arr[row][column]
            if not missing_record_list:
                missing_record = "N/A"
            else:
                # map timer record (float) into string, form: t0: 1.5s, t1: 2.3s...
                # only map most recent 4 times to avoid overcrowding div cards
                record_len = len(missing_record_list)
                start = 0 if record_len < 4 else (record_len-4)
                missing_record = map(
                    lambda x: "t{}: {}s".format(x, missing_record_list[x]), range(start, record_len)
                )
                # turn map obj into list
                missing_record = list(missing_record)
                # show the latest missing time first
                missing_record.reverse()
                # list of strings to single string ( joined by ", ")
                missing_record = ", ".join(missing_record)

            behaelter_dict[dict_key_var] = {
                # coords = ymin, xmin or "N/A"
                "coords": coords,
                "score": score,
                "filled": filled,
                "wkstk_score": wkstk_score,
                "color": color,
                "missing": missing,
                "missing_record": missing_record
            }

            dict_key_var += 1
    return behaelter_dict


hochregallager = Hochregallager()
cam_resolution = (1280, 720)

if __name__ == "__main__":
    # app.run(debug=True)
    app.run()
