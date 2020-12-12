import csv

from aisdata import ship
from aisdata import aisstate
from aisdata.anomaly import Anomaly
from aisdata import route
from aisdata import state
from aisdata import scope

def read_float(row, field, default):
    try:
        return float(row[field])
    except Exception:
        return default

class _ImportedScope(scope.Scope):

    def __init__(self, longitude_min_user, longitude_max_user, latitude_min_user, latitude_max_user, t_min, t_max,
                 longitude_min=None, longitude_max=None, latitude_min=None, latitude_max=None):
        scope.Scope.__init__(self, longitude_min_user, longitude_max_user, latitude_min_user, latitude_max_user, t_min, t_max,
                 longitude_min, longitude_max, latitude_min, latitude_max)
        self.ship_dict = {}

class _ImportedShip(ship.Ship):

    def __init__(self, sourcemmsi, shipname, shiptype, length, width):
        ship.Ship.__init__(self, sourcemmsi, shipname, shiptype, length, width)
        self.route_dict = {}

class Reader:

    """Reads stored data from a directory."""

    def __init__(self, directory):
        self.directory = directory

    def read(self):
        """Reads files and returns scope with content of the files."""
        self.scope = self._read_scope()
        self._read_ships()
        self._read_ais_states()
        self._read_routes()
        self._read_states("states.csv", lambda route: route.states)
        self._read_states("predicted_states.csv", lambda route: route.predicted_states)
        self._read_anomalies()
        number = lambda a: a.number_in_route
        for s in self.scope.ship_dict.values():
            s.routes = s.route_dict.values()
            for r in s.routes:
                r.states.sort(key=number)
        self.scope.ships = self.scope.ship_dict.values()
        return self.scope

    def _read_scope(self):
        with open(self.directory + "/scope.csv") as inputfile:
            reader = csv.DictReader(inputfile, delimiter=',')
            for row in reader:
                lon_min = float(row["lon_min"])
                lon_max = float(row["lon_max"])
                lat_min = float(row["lat_min"])
                lat_max = float(row["lat_max"])
                user_lon_min = read_float(row, "user_lon_min", lon_min)
                user_lon_max = read_float(row, "user_lon_max", lon_max)
                user_lat_min = read_float(row, "user_lat_min", lat_min)
                user_lat_max = read_float(row, "user_lat_max", lat_max)
                t_min = float(row["t_min"])
                t_max = float(row["t_max"])
                return _ImportedScope(lon_min, lon_max, lat_min, lat_max, t_min, t_max, user_lon_min,
                                      user_lon_max, user_lat_min, user_lat_max)

    def _read_ships(self):
        with open(self.directory + "/ships.csv") as inputfile:
            reader = csv.DictReader(inputfile, delimiter=',')
            for row in reader:
                sourcemmsi = row["sourcemmsi"]
                shipname = row["shipname"]
                shiptype = int(row["shiptype"])
                length = float(row["length"])
                width = float(row["width"])
                self.scope.ship_dict[sourcemmsi] = \
                        _ImportedShip(sourcemmsi, shipname, shiptype, length, width)

    def _read_ais_states(self):
        with open(self.directory + "/ais_states.csv") as inputfile:
            reader = csv.DictReader(inputfile, delimiter=',')
            for row in reader:
                sourcemmsi = row["sourcemmsi"]
                lon = float(row["lon"])
                lat = float(row["lat"])
                navigationalstatus = int(row["navigationalstatus"])
                rateofturn = float(row["rateofturn"])
                speedoverground = float(row["speedoverground"])
                courseoverground = float(row["courseoverground"])
                trueheading = float(row["trueheading"])
                timestamp = float(row["timestamp"])
                destination = row["destination"]
                eta = int(row["eta"])
                draught = float(row["draught"])
                self.scope.ship_dict[sourcemmsi].ais_states.append(
                    aisstate.AISState(lon, lat, navigationalstatus, rateofturn,
                    speedoverground, courseoverground, trueheading, timestamp, destination, eta, draught))

    def _read_routes(self):
        with open(self.directory + "/routes.csv") as inputfile:
            reader = csv.DictReader(inputfile, delimiter=',')
            for row in reader:
                sourcemmsi = row["sourcemmsi"]
                destination = row["destination"]
                eta = int(row["eta"])
                t_begin = int(row["t_begin"])
                t_end = int(row["t_end"])
                self.scope.ship_dict[sourcemmsi].route_dict[t_begin] = \
                        route.Route(destination, eta, t_begin, t_end)

    def _read_states(self, file, select_state):
        with open(self.directory + "/" + file) as inputfile:
            reader = csv.DictReader(inputfile, delimiter=',')
            for row in reader:
                sourcemmsi = row["sourcemmsi"]
                t_begin = int(row["t_begin"])
                nr = int(row["number_in_route"])
                navigationalstatus = int(row["navigationalstatus"])
                rateofturn = float(row["rateofturn"])
                speedoverground = float(row["speedoverground"])
                courseoverground = float(row["courseoverground"])
                trueheading = float(row["trueheading"])
                lon = float(row["lon"])
                lat = float(row["lat"])
                t_day = float(row["t_day"])
                t_week = float(row["t_week"])
                draught = float(row["draught"])
                r = self.scope.ship_dict[sourcemmsi].route_dict[t_begin]
                select_state(r).append(
                    state.State(nr, lon, lat, navigationalstatus, rateofturn,
                                speedoverground, courseoverground, trueheading, t_day, t_week, draught))

    def _read_anomalies(self):
        with open(self.directory + "/anomalies.csv") as inputfile:
            reader = csv.DictReader(inputfile, delimiter=',')
            for row in reader:
                sourcemmsi = row["sourcemmsi"]
                t_begin = int(row["t_begin"])
                nr = int(row["number_in_route"])
                error = float(row["error"])
                anomaly = float(row["anomaly"])
                r = self.scope.ship_dict[sourcemmsi].route_dict[t_begin]
                r.anomalies.append(Anomaly(nr, error, anomaly))
