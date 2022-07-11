import time
import coordinates as coord
import werk_stueck
import behaelter

class Hochregallager:
    def __init__(self):
        self.behaelter_obj_list = []
        self.width_in_px = None
        self.height_in_px = None
        self.coordinates = None
        self.behaelter_arr = [[None for x in range(3)] for y in range(3)]
        self.grid_cell_timer_arr = [[0 for x in range(3)] for y in range(3)]
        self.grid_successfully_initialized = False
        self.image = None

    def set_image(self, image):
        self.image = image

    def add_behaelter(self, behaelter_obj):
        self.behaelter_obj_list.append(behaelter_obj)

    def assign_grid_pos(self, behaelter, row, column):
        self.behaelter_arr[row][column] = behaelter
        # if behaelter.empty:
        #     print('POS:{}x{} is EMPTY')
        # else:
        #     print('POS:{}x{} is {}'.format(
        #         row, column, behaelter.werk_stueck.color))

    def check_for_missing_behaelter(self):
        # check if a behaelter was removed, if so - call remove
        pass

    def remove_behaelter(self, row, column):
        self.behaelter_arr[row][column] = None
        # -> TIMER

    # call after a new frame is presented

    def clear_behaelter_list(self):
        self.behaelter_obj_list = []

    def initialize_grid_coordinates(self):
        # coordinates should only be None before the first frame, grid_successfully_initialized being false
        # means (at least) one Behaelter was missing in the frame(s) before, thereby distorting the coordinates
        if not self.grid_successfully_initialized:
            self.coordinates = coord.get_approx_hochregallager_grid_coordinates(
                self)
            self.width_in_px = coord.get_approx_hochregallager_grid_width(
                self)
            self.height_in_px = coord.get_approx_hochregallager_grid_height(
                self)
            if len(self.behaelter_obj_list) == 9:
                self.grid_successfully_initialized = True

    def start_grid_cell_timer(self, row, column, current_time):
        self.grid_cell_timer_arr[row][column] = current_time
        # print("TIMER WAS STARTED")

    def stop_grid_cell_timer(self, row, column):
        self.grid_cell_timer_arr[row][column] = 0

    def get_behaelter_pos_by_behaelter_obj(self, behaelter_obj):
        # grid shape: 3x3
        for row in range(3):
            for column in range(3):
                if behaelter_obj == self.behaelter_arr[row][column]:
                    return (row, column)
        return None

    def get_behaelter_pos_by_behaelter_box(self, image, behaelter_box):
        im_height = image.shape[0]
        im_width = image.shape[1]
        # behaelter_arr contains behaelter obj with de'normalized' coordinates
        behaelter_box = coord.get_box_coordinates_from_normalized_coordinates(
            behaelter_box, im_height, im_width)
        # grid shape: 3x3
        for row in range(3):
            for column in range(3):
                # REMOVE following line
                if self.behaelter_arr[row][column] is None:
                    pass

                else:
                    if behaelter_box == self.behaelter_arr[row][column].bounding_box:
                        return (row, column)

        return None

    def get_behaelter_obj_by_behaelter_box(self, image, behaelter_box):
        im_height = image.shape[0]
        im_width = image.shape[1]
        # behaelter_arr contains behaelter obj with de'normalized' coordinates
        behaelter_box = coord.get_box_coordinates_from_normalized_coordinates(
            behaelter_box, im_height, im_width)
        # grid shape: 3x3
        for behaelter_obj in self.behaelter_obj_list:
            if behaelter_obj.bounding_box == behaelter_box:
                return behaelter_obj
        return None

    def get_werkstueck_pos_by_werkstueck_box(self, image, wkstk_box):
        im_height = image.shape[0]
        im_width = image.shape[1]
        # behaelter_arr contains behaelter obj with de'normalized' coordinates
        wkstk_box = coord.get_box_coordinates_from_normalized_coordinates(
            wkstk_box, im_height, im_width)
        # grid shape: 3x3
        for row in range(3):
            for column in range(3):
                if wkstk_box == self.behaelter_arr[row][column].werk_stueck.bounding_box:
                    return (row, column)
        return None

    # timer runtime in sec
    def get_grid_cell_timer_value(self, row, column):
        time_diff = time.time() - self.grid_cell_timer_arr[row][column]
        return round(time_diff, 2)
