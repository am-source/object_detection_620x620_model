import coordinates as coord
from werk_stueck import WerkStueck
import color_detector

class Behaelter:
    def __init__(self, image, behaelter_box, behaelter_score, filtered_WerkStueck_detections):
        im_height = image.shape[0]
        im_width = image.shape[1]

        wkst_boxes, _, wkst_scores = filtered_WerkStueck_detections

        # ymin, xmin, ymax, xmax
        self.bounding_box = coord.get_box_coordinates_from_normalized_coordinates(
            behaelter_box, im_height, im_width)
        self.score = behaelter_score
        self.werk_stueck = self.has_WerkStueck(
            wkst_boxes, wkst_scores, image, im_height, im_width)
        self.empty = True if self.werk_stueck is None else False

    def has_WerkStueck(self, wkst_boxes, wkst_scores, image, im_height, im_width):
        return get_werkstueck_contained_in_behaelter(self.bounding_box, wkst_boxes, wkst_scores, image, im_height, im_width)


def get_werkstueck_contained_in_behaelter(
    behaelter_box, wkst_boxes, wkst_scores, image, im_height, im_width
):
    intersect_elements = coord.bounding_box_intersect(
        behaelter_box, wkst_boxes, wkst_scores, im_height=im_height, im_width=im_width, needs_normalization=True)

    if len(intersect_elements) == 1:
        wkstk_box = intersect_elements[0][0]
        wkstk_score = intersect_elements[0][1]
        wkstk_color = color_detector.detect_color_in_bounding_box(image, wkstk_box, False)

        wkstk = WerkStueck(wkstk_box, wkstk_score, wkstk_color)
        return wkstk

    else:
        if len(intersect_elements) > 1:
            print(
                "{} wkstk elements intersect behaelter:{}".format(
                    len(intersect_elements), behaelter_box
                )
            )
        return None
