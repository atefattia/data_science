"""Dynamic AIS data and the static AIS data changing over time."""


class AISState:

    """One AISState object represents the AIS data of one time stamp."""

    def __init__(self, longitude, latitude, status, rate_of_turn,
                 speed_over_ground, course_over_ground, true_heading, timestamp,
                 destination, eta, draught):
        """Constructor of AISState

        Parameters:
            longitude: GPS-Coordinates (double)
            latitude: GPS-Coordinates (double)
            status: navigational status (integer)
            rate_of_turn: degree per min(double)
            speed_over_ground: knots (double)
            course_over_ground: degree (double)
            true_heading: degree (double)
            timestamp: Unix timestamp
            destination: Destination from the last static AIS data of the same ship (string)
            eta: ETA from the last static AIS data of the same ship (UTC string, MM-DD-HH-MM)
            draught: Draught from the last static AIS data of the same ship
        """
        self.longitude = longitude
        self.latitude = latitude
        self.status = status
        self.rate_of_turn = rate_of_turn
        self.speed_over_ground = speed_over_ground
        self.course_over_ground = course_over_ground
        self.true_heading = true_heading
        self.timestamp = timestamp
        self.destination = destination
        self.eta = eta
        self.draught = draught
