import time
import coordinates as coord
import werk_stueck
import behaelter

class Hochregallager:
    def __init__(self):
        # list of all behaelter objects created, not yet assigned positions
        self.behaelter_obj_list = []
        self.width_in_px = None
        self.height_in_px = None
        self.coordinates = None
        # list of HRL positions, either assigned None or a behealer object
        self.behaelter_arr = [[None for x in range(3)] for y in range(3)]
        # current missing timer for all positions, 0 when not activated, otherwise time.time() when removed
        self.grid_cell_timer_arr = [[0 for x in range(3)] for y in range(3)]
        # missing_time_arr keeps a list of missing times of Behaelter in every cell
        self.missing_time_record_arr = [[[] for x in range(3)] for y in range(3)]
        self.grid_successfully_initialized = False
        self.image = None

    # handles all grid related tasks
    def handle_grid_coordinates_and_pos_assignment(self):
        self.initialize_grid_coordinates()
        coord.handle_grid_positions(self)

    def initialize_grid_coordinates(self):
        approx_coord = coord.get_approx_hochregallager_grid_coordinates(
            self)
        if approx_coord is not None:
            self.coordinates = approx_coord
            self.width_in_px = coord.get_approx_hochregallager_grid_width(
                self)
            self.height_in_px = coord.get_approx_hochregallager_grid_height(
                self)

    def set_image(self, image):
        # assign current frame to HRL
        self.image = image

    def add_behaelter(self, behaelter_obj):
        self.behaelter_obj_list.append(behaelter_obj)

    def assign_grid_pos(self, behaelter, row, column):
        self.behaelter_arr[row][column] = behaelter

    def remove_behaelter(self, row, column):
        self.behaelter_arr[row][column] = None

    # called after a new frame is presented
    def clear_behaelter_list(self):
        self.behaelter_obj_list = []

    def start_grid_cell_timer(self, row, column, current_time):
        self.grid_cell_timer_arr[row][column] = current_time

    def stop_grid_cell_timer(self, row, column):
        missing_time = self.get_grid_cell_timer_value(row, column)
        # at least 5s to be counted as missing, since missing times of around 0-3 are usually mistakes resulting
        # from detection and processing or loss of frames
        if missing_time >= 5:
            self.missing_time_record_arr[row][column].append(missing_time)
        # print("TIME RECORD OF {}x{}: {}".format(row, column, self.missing_time_record_arr[row][column]))
        self.grid_cell_timer_arr[row][column] = 0

    def get_behaelter_pos_by_behaelter_obj(self, behaelter_obj):
        # grid shape: 3x3
        for row in range(3):
            for column in range(3):
                if behaelter_obj == self.behaelter_arr[row][column]:
                    return (row, column)
        return None

    def get_behaelter_pos_by_behaelter_box(self, behaelter_box):
        # grid shape: 3x3
        for row in range(3):
            for column in range(3):
                # check None first to avoid attributeError
                if self.behaelter_arr[row][column] is not None:
                    if behaelter_box == self.behaelter_arr[row][column].bounding_box:
                        return (row, column)
        return None

    def get_behaelter_obj_by_behaelter_box(self, behaelter_box):
        # grid shape: 3x3
        for behaelter_obj in self.behaelter_obj_list:
            if behaelter_obj.bounding_box == behaelter_box:
                return behaelter_obj
        return None

    def get_werkstueck_pos_by_werkstueck_box(self, wkstk_box):
        # grid shape: 3x3
        for row in range(3):
            for column in range(3):
                if self.behaelter_arr[row][column] is not None:
                    if wkstk_box == self.behaelter_arr[row][column].werk_stueck.bounding_box:
                        return (row, column)
        return None

    # timer runtime in sec
    def get_grid_cell_timer_value(self, row, column):
        time_diff = time.time() - self.grid_cell_timer_arr[row][column]
        return round(time_diff, 2)
