"""Time horizont and location for observing ship routes."""


class Scope:

    """A given time and location window to observe ship routes. The window has
    to be small enought to use a flat world assumption."""

    def __init__(self, longitude_min, longitude_max, latitude_min, latitude_max, t_min, t_max,
            longitude_min_user=None, longitude_max_user=None, latitude_min_user=None, latitude_max_user=None):
        """Constructor of Scope

        Parameters:
            longitude_min: most western longitude of the ships
            longitude_max: most eastern longitude of the ships
            latitude_min: most southern latitude of the ships
            latitude_max: most northern latitude of the ships
            t_min: Start of time horizont (Unix timestamp)
            t_max: end of time horizont (Unix timestamp)
        optional Parameters (default is same as defined before):
            longitude_min_user: most western longitude given from the user
            longitude_max_user: most eastern longitude given from the user
            latitude_min_user: most southern latitude given from the user
            latitude_max_user: most northern latitude given from the user
        """

        # given from the user
        self.longitude_min = longitude_min
        self.longitude_max = longitude_max
        self.latitude_min = latitude_min
        self.latitude_max = latitude_max

        if longitude_min_user is not None:
            self.longitude_min_user = longitude_min_user
            self.longitude_max_user = longitude_max_user
            self.latitude_min_user = latitude_min_user
            self.latitude_max_user = latitude_max_user
        else:
            # actual max min of the available ships
            self.longitude_min_user = longitude_min
            self.longitude_max_user = longitude_max
            self.latitude_min_user = latitude_min
            self.latitude_max_user = latitude_max

        self.t_min = t_min
        self.t_max = t_max
        self.ships = []

    def getRealValues(self, state):
        """ returns the longitude and latitude values before the normalization
        return: denormalized longitude and latitude
        """
        longitude_diff = self.longitude_max - self.longitude_min
        latitude_diff = self.latitude_max - self.latitude_min

        longitude = (state.longitude * longitude_diff) + self.longitude_min
        latitude = (state.latitude * latitude_diff) + self.latitude_min

        return longitude, latitude
