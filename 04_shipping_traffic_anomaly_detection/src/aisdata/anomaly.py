"""Anomaly of a ship state."""


class Anomaly:

    """..."""

    def __init__(self, number_in_route, error, anomaly):
        """Constructor of anomaly

        Parameters:
            errors: distance error between state and predicted state
            anomaly: anomaly classification of state [0-1]
        """
        self.number_in_route = number_in_route
        self.error = error
        self.anomaly = anomaly
