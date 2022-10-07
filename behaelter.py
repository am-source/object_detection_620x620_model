import coordinates as coord
from werk_stueck import WerkStueck
import color_detector


class Behaelter:
    def __init__(self, image, behaelter_box, behaelter_score, filtered_WerkStueck_detections):

        wkst_boxes, _, wkst_scores = filtered_WerkStueck_detections

        # ymin, xmin, ymax, xmax
        self.bounding_box = behaelter_box
        self.score = behaelter_score
        self.werk_stueck = self.has_WerkStueck(
            wkst_boxes, wkst_scores, image)
        self.empty = True if self.werk_stueck is None else False

    def has_WerkStueck(self, wkst_boxes, wkst_scores, image):
        return get_werkstueck_contained_in_behaelter(self.bounding_box, wkst_boxes, wkst_scores, image)


def get_werkstueck_contained_in_behaelter(
    behaelter_box, wkst_boxes, wkst_scores, image
):
    intersect_elements = coord.bounding_box_intersect(
        behaelter_box, wkst_boxes, wkst_scores)

    if len(intersect_elements) == 1:
        wkstk_box = intersect_elements[0][0]
        wkstk_score = intersect_elements[0][1]
        wkstk_color = color_detector.detect_color_in_bounding_box(image, wkstk_box)

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
