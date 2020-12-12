"""State of a ship based on dynamic AIS data"""

import math

draught_max = 25.5  # meters

class State:

    """State of a ship at a point in time, based on AIS data, normalized to a scope."""

    def __init__(
            self, number_in_route, longitude, latitude, status, rate_of_turn, speed_over_ground,
            course_over_ground, true_heading, t_day, t_week, draught):
        """Constructor of State

        Parameters:
            longitude: longitude inside the scope [0-1]
            latitude: latitude inside the scope [0-1]
            status: status of the ship (integer)
            rate_of_turn: [0-1]
            speed_over_ground: [0-1]
            course_over_ground: [0-1]
            true_heading: [0-1]
            t_day: [0-1]
            t_week: {0, 1, 2, 3, 4, 5, 6} (integer)
            draught: normalized to draught_max [0-1]
        """
        self.number_in_route = number_in_route
        self.longitude = longitude
        self.latitude = latitude
        self.status = status
        self.statusarray = []
        self.rate_of_turn = rate_of_turn
        self.valid_rate_of_turn = True
        self.speed_over_ground = speed_over_ground
        self.valid_speed_over_ground = True
        self.course_over_ground = course_over_ground
        self.valid_course_over_ground = True
        self.true_heading = true_heading
        self.valid_true_heading = True
        self.t_day = t_day
        self.t_week = t_week
        self.t_weekarray = []
        self.draught = draught

    def distance(self, other):
        """ calculates the spatial distance between two states
        Paramters:
            other: the other state
        """
        return math.sqrt((self.longitude - other.longitude) ** 2 + \
                         (self.latitude - other.latitude) ** 2)
