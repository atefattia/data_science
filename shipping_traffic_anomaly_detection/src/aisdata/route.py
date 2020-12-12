"""Time horizont of a ship with a consistent behaviour."""

import itertools

class Route:

    """A route represents the time horizont in which the ship follows the same purpose
    eg. arriving one destination, going fishing."""

    # time intervall between two states (in seconds)
    TIME_STEP = 30

    def __init__(self, destination, eta, t_begin, t_end):
        """Constructor of Route

        Parameters:
            destination: (string)
            eta: (Unix timestamp)
            t_begin: (Unix timestamp)
            t_end: (Unix timestamp)
        """
        self.destination = destination
        self.destinationarray = []  # TODO
        self.eta = eta
        self.valid_eta = True
        self.t_begin = t_begin
        self.t_end = t_end
        self.states = []
        self.predicted_states = []
        self.anomalies = []

    def __is_in_time_span(self, t):
        return self.t_begin < t and t < self.t_end

    def overlapping_in_time(self, other):
        """ whether two routes are overlapping in time
        Paramterers:
            other the other route
        """
        return self.__is_in_time_span(other.t_begin) or \
               self.__is_in_time_span(other.t_end)

    def distance_to_route(self, other):
        """ returns the distance between two routes. The distance between two routes is defined as the distance between the two closes points of the routes.
        Paramters:
            other: the other route
        """
        return min(map(lambda t: t[0].distance(t[1]), itertools.product(self.states, other.states))) 
