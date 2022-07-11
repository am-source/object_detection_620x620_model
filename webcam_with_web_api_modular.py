#!/usr/bin/env python
# coding: utf-8
import cv2
import detect_handler as detect
from flask import Flask, render_template, Response, jsonify
from hoch_regal_lager import Hochregallager
from camera import MyCamera
import coordinates


app = Flask(__name__)
hochregallager = Hochregallager()

if __name__ == "__main__":
    # app.run(debug=True)
    app.run()


########## FLASK ROUTES ############################################################################
def generate_frames():
    cam = MyCamera()
    print("ENTERED generate_frames!")
    while True:
        print("IN while True BLOCK!")
        if not cam.frame_available():
            continue
        print("IN while True BLOCK! - IF ret WAS TRUE")

        image_np_with_detections = detect.handle_detection(camera=cam, hochregallager=hochregallager)

        # encode for Response stream
        ret, buffer = cv2.imencode('.jpg', image_np_with_detections)
        image_np_with_detections = buffer.tobytes()

        yield(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image_np_with_detections + b'\r\n'
        )


@app.route('/_stuff', methods=['GET'])
def stuff():
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
                # in case the behaelter hasn't been detected yet (for the first time)
                if hochregallager.grid_cell_timer_arr[row][
                    column] == 0 and not hochregallager.grid_successfully_initialized:
                    missing = "N/A"
                else:
                    missing = str(hochregallager.get_grid_cell_timer_value(row, column))

            else:
                behaelter = hochregallager.behaelter_arr[row][column]
                # param hochregallager.image here is for img width and height
                ymin, xmin, ymax, xmax = coordinates.get_box_coord_relative_to_grid_coord(hochregallager.image,
                                                                              behaelter.bounding_box, hochregallager)
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
    return behaelter_dict
