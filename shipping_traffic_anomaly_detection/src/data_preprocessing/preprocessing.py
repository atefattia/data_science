import math
import sys
from aisdata import route
from aisdata import state
from scipy import interpolate
from datetime import datetime
from datetime import date
import ephem
import numpy as np

""" Data cleaning and preprocessing for the RNN """


class Preprocessing:

    # defines the allowed ship's type
    FISHING = 30
    PASSENGER = range(60, 69)
    CARGO = range(70, 79)
    TANKER = range(80, 89)

    # defines the shortest possible sequence in seconds (set to 3min = 3 * 60sec)
    MIN_SEQUENCE_SECONDS = 3 * 60

    # defines the shortest possible sequence number for a ship
    MIN_AIS_STATES_NUMBER = 10

    # defines the minimum speed to check if a ship is moving or not (0.6 knot = 1.112 km/h)
    MIN_SPEED_KNOTS = 0.6

    # defines the max number of NaN values for a ship
    MAX_NAN_ENTRIES = 3

    # defines the max time span between two ais_states in one route (set to 1h = 3600sec)
    MAX_TIMESPAN_AIS_STATES = 3600

    # defines the minimal number of states in a route
    MIN_STATES_PER_ROUTE = 10

    # Dictionary to one hot encode destinations
    all_destination = {}

    # Dictionary to one hot encode status
    all_status = {}

    def __init__(self, vscope, ships):
        """Constructor of Preprocessing

        Parameters:
            vscope: geographical and temporally delimiter for the data (aisdata.scope)
            ships: list of ships imported with data_import.aisimport
        """
        self.scope = vscope
        self.ships = ships
        self.all_destination.clear()
        self.all_status.clear()

    def __shipInScope(self, state):
        """ Checks if the given state is within the scope

        Parameters:
        state: the ship's state to check
        return: True if state is within the scope, otherwise False
        """
        if (state.longitude >= self.scope.longitude_min and state.longitude <= self.scope.longitude_max and state.latitude >= self.scope.latitude_min and state.latitude <= self.scope.latitude_max and state.timestamp >= self.scope.t_min and state.timestamp <= self.scope.t_max):
            return True
        return False

    def filter_by_scope(self):
        """ Removes states outside the given scope. After removing states, ships without remaining states will be deleted
        """
        for ship in self.ships:
            ship.ais_states[:] = [x for x in ship.ais_states if self.__shipInScope(x)]
        self.ships[:] = [x for x in self.ships if (len(x.ais_states) > self.MIN_AIS_STATES_NUMBER)]

    def __isShipTypeCorrect(self, ship):
        """ Checks if the given ship's type is within the defined types

        Parameters:
        ship: the ship to be checked
        return: True if the shiptype is FISCHING, PASSENGER, CARGO or TANKER, otherwise False
        """
        if (ship.shiptype == self.FISHING or ship.shiptype in self.PASSENGER or ship.shiptype in self.CARGO or ship.shiptype in self.TANKER):
            return True
        return False

    def filter_by_shiptype(self):
        """ Removes ships, which do not have the required type
        """
        self.ships[:] = [x for x in self.ships if self.__isShipTypeCorrect(x)]

    def delete_not_moving_ships(self):
        """ Eliminates ships, which do not move
        """
        for ship in self.ships:
            ship.ais_states[:] = [x for x in ship.ais_states if (x.speed_over_ground > self.MIN_SPEED_KNOTS)]
        self.ships[:] = [x for x in self.ships if (len(x.ais_states) > self.MIN_AIS_STATES_NUMBER)]

    def __isSequenceNotShort(self, ship):
        """ Checks if the given ship has sent signals in a larger period than MIN_SEQUENCE_SECONDS

        Parameters:
        ship: the ship to be checked
        return: True if the sequence is considered to be short, otherwise False
        """
        t_max = t_min = 0
        if (len(ship.ais_states) > 0):
            t_max = ship.ais_states[0].timestamp
            t_min = ship.ais_states[0].timestamp
            for vState in ship.ais_states:
                if vState.timestamp > t_max:
                    t_max = vState.timestamp
                if vState.timestamp < t_min:
                    t_min = vState.timestamp
        if (t_max - t_min >= self.MIN_SEQUENCE_SECONDS):
            return True
        return False

    def delete_short_sequences(self):
        """ Removes ships, which have send signals in a period less than a defined value
        (in sec) MIN_SEQUENCE_SECONDS
        """
        self.ships[:] = [x for x in self.ships if self.__isSequenceNotShort(x)]
        self.ships[:] = [x for x in self.ships if (len(x.ais_states) > self.MIN_AIS_STATES_NUMBER)]

    def __IsNaNAdmissible(self, state, count_NaN):
        """ Count the number of NaN entries for a ship

        Parameters:
        state: the state of the ship to be checked
        count_NaN: number of NaN entries for a ship
        return: the number of NaN values
        """
        if (math.isnan(state.rate_of_turn)):
            count_NaN += 1
        if (math.isnan(state.draught)):
            count_NaN += 1
        if (state.destination == ""):
            count_NaN += 1
        return count_NaN

    def filter_by_NaN(self):
        """ Removes ships, which have more NaN entries than permitted MAX_NAN_ENTRIES
        """
        for ship in self.ships:
            count_NaN = 0
            if (math.isnan(ship.length)):
                count_NaN += 1
            if (math.isnan(ship.width)):
                count_NaN += 1
            ship.ais_states[:] = [x for x in ship.ais_states if (self.__IsNaNAdmissible(x, count_NaN) <= self.MAX_NAN_ENTRIES)]
        self.ships[:] = [x for x in self.ships if (len(x.ais_states) > self.MIN_AIS_STATES_NUMBER)]

    def __meanLength(self, shiptype):
        """ Calculates the mean length for ships with the given shiptype

        Parameters:
        shiptype: the shiptype to be considered to cumpute the mean
        return: the mean length
        """
        ship_count = 0
        total_length = 0
        for ship in self.ships:
            if (ship.shiptype == shiptype):
                if not (math.isnan(ship.length)):
                    ship_count += 1
                    total_length += ship.length
        if (ship_count == 0):
            mean = 0
        else:
            mean = total_length / ship_count
        return mean

    def __meanWidth(self, shiptype):
        """ Calculates the mean width for ships with the given shiptype

        Parameters:
        shiptype: the shiptype to be considered to cumpute the mean
        return: the mean width
        """
        ship_count = 0
        total_width = 0
        for ship in self.ships:
            if (ship.shiptype == shiptype):
                if not (math.isnan(ship.width)):
                    ship_count += 1
                    total_width += ship.width
        if (ship_count == 0):
            mean = 0
        else:
            mean = total_width / ship_count
        return mean

    def __meanDraught(self, shiptype):
        """ Calculates the mean draught for ships with the given shiptype

        Parameters:
        shiptype: the shiptype to be considered to cumpute the mean
        return: the mean draught
        """
        ship_count = 0
        total_draught = 0
        for ship in self.ships:
            if (ship.shiptype == shiptype):
                if (len(ship.ais_states) > 0):
                    if not (math.isnan(ship.ais_states[0].draught)):
                        ship_count += 1
                        total_draught += ship.ais_states[0].draught
        if (ship_count == 0):
            mean = 0
        else:
            mean = total_draught / ship_count
        return mean

    def handle_missing_values(self):
        """ Handles missing values like NaN or empty strings
        """
        for ship in self.ships:
            if (ship.shipname == ""):
                ship.shipname = "not_available"  # no name available
            if (math.isnan(ship.length)):
                ship.length = self.__meanLength(ship.shiptype)  # calculates the mean length taking into account the shiptype
            if (math.isnan(ship.width)):
                ship.width = self.__meanWidth(ship.shiptype)  # calculates the mean width taking into account the shiptype
            for vState in ship.ais_states:
                if (math.isnan(vState.rate_of_turn)):
                    vState.rate_of_turn = -128  # no turn information available (default)
                if (vState.destination == ""):
                    vState.destination = "not_available"  # no destination available
                if (math.isnan(vState.draught)):
                    vState.draught = self.__meanDraught(ship.shiptype)  # calculates the mean draught taking into account the shiptype

    def _isDay(self, vlong, lat, time):
        """ Checks at a given longitude, latitude and time, if the sun is up (day or night)

        Parameters:
        vlong: the given longitude
        lat: the given latitude
        time: the given time
        return: 1 if it is a day, otherwise 0
        """
        isDay = 1.0
        observer = ephem.Observer()
        observer.long = str(vlong)
        observer.lat = str(lat)
        observer.date = time
        sun = ephem.Sun()
        sun.compute(observer)
        current_sun_alt = sun.alt
        if ((current_sun_alt * 180) / math.pi < -12):
            isDay = 0.0
        return isDay

    def _weekDay(self, time):
        """ Checks at a given time which week day it is

        Parameters:
        time: the given time
        return: 0 if monday, 1 if tuesday and so on
        """
        year, month, day = (int(x) for x in time.split('-'))
        week_day = date(year, month, day).weekday()
        return week_day

    def __clearLists(self, timestamp_list, long_list, lat_list, RoT_list, SoG_list, CoG_list, Th_list):
        """ empty all given lists
        """
        timestamp_list.clear()
        long_list.clear()
        lat_list.clear()
        RoT_list.clear()
        SoG_list.clear()
        CoG_list.clear()
        Th_list.clear()

    def create_routes(self):
        """ creates routes for ships for each destination and interpolates entries by time to make states equidistant
        """
        number_in_route = 0
        timestamp_list = []
        long_list = []
        lat_list = []
        long_list = []
        RoT_list = []
        SoG_list = []
        CoG_list = []
        Th_list = []
        for ship in self.ships:
            self.__clearLists(timestamp_list, long_list, lat_list, RoT_list, SoG_list, CoG_list, Th_list)
            if (len(ship.ais_states) != 0):
                destination = ship.ais_states[0].destination
                eta = ship.ais_states[0].eta
                # check with t_prev if the next ais_states was received before MAX_TIMESPAN_AIS_STATES
                t_prev = t_begin = t_end = ship.ais_states[0].timestamp
                # create route with destination
                vRoute = route.Route(destination, eta, t_begin, t_end)
                for vState in ship.ais_states:
                    # map the destination and status to an index for one hot encoding
                    if (destination not in self.all_destination):
                        self.all_destination[destination] = len(self.all_destination)
                    if (vState.status not in self.all_status):
                        self.all_status[vState.status] = len(self.all_status)
                    # create one route if destination same and time span between successive ais_states less than 1 hour, otherwise create new one
                    if (vState.destination == destination and (vState.timestamp - t_prev) < self.MAX_TIMESPAN_AIS_STATES):
                        t_prev = vState.timestamp
                        timestamp_list.append(vState.timestamp)
                        long_list.append(vState.longitude)
                        lat_list.append(vState.latitude)
                        status = vState.status
                        RoT_list.append(vState.rate_of_turn)
                        SoG_list.append(vState.speed_over_ground)
                        CoG_list.append(vState.course_over_ground)
                        Th_list.append(vState.true_heading)
                        draught = vState.draught
                    else:
                        n = (timestamp_list[-1] - timestamp_list[0]) // vRoute.TIME_STEP
                        if (len(timestamp_list) > 3 and n >= self.MIN_STATES_PER_ROUTE):
                            step = t_begin
                            long_interpolator = interpolate.interp1d(timestamp_list, long_list, kind='linear')
                            lat_interpolator = interpolate.interp1d(timestamp_list, lat_list, kind='linear')
                            RoT_interpolator = interpolate.interp1d(timestamp_list, RoT_list, kind='nearest')
                            SoG_interpolator = interpolate.interp1d(timestamp_list, SoG_list, kind='linear')
                            CoG_interpolator = interpolate.interp1d(timestamp_list, CoG_list, kind='linear')
                            Th_interpolator = interpolate.interp1d(timestamp_list, Th_list, kind='linear')
                            while (step < timestamp_list[-1]):
                                longitude = long_interpolator(step)
                                latitude = lat_interpolator(step)
                                rate_of_turn = RoT_interpolator(step)
                                speed_over_ground = SoG_interpolator(step)
                                course_over_ground = CoG_interpolator(step)
                                true_heading = Th_interpolator(step)
                                utc_step = datetime.utcfromtimestamp(step).strftime('%Y-%m-%d %H:%M:%S')
                                # day: 1 and night: 0
                                t_day = self._isDay(longitude, latitude, utc_step)
                                # Monday: 0 - Tuesday: 1 - Wednesday: 2, Thursday: 3 - Friday: 4 - Saturday: 5 - Sunday: 6
                                t_week = self._weekDay(datetime.utcfromtimestamp(step).strftime('%Y-%m-%d'))
                                vRoute.states.append(state.State(number_in_route, longitude, latitude, status, rate_of_turn, speed_over_ground, course_over_ground, true_heading, t_day, t_week, draught))
                                number_in_route += 1
                                step += vRoute.TIME_STEP
                            vRoute.t_end = step - vRoute.TIME_STEP
                            ship.routes.append(vRoute)
                        destination = vState.destination
                        eta = vState.eta
                        t_prev = t_begin = t_end = vState.timestamp
                        vRoute = route.Route(destination, eta, t_begin, t_end)
                        self.__clearLists(timestamp_list, long_list, lat_list, RoT_list, SoG_list, CoG_list, Th_list)
                        timestamp_list.append(vState.timestamp)
                        long_list.append(vState.longitude)
                        lat_list.append(vState.latitude)
                        RoT_list.append(vState.rate_of_turn)
                        SoG_list.append(vState.speed_over_ground)
                        CoG_list.append(vState.course_over_ground)
                        Th_list.append(vState.true_heading)
                n = (timestamp_list[-1] - timestamp_list[0]) // vRoute.TIME_STEP
                if (len(timestamp_list) > 3 and n >= self.MIN_STATES_PER_ROUTE):
                    step = t_begin
                    long_interpolator = interpolate.interp1d(timestamp_list, long_list, kind='linear')
                    lat_interpolator = interpolate.interp1d(timestamp_list, lat_list, kind='linear')
                    RoT_interpolator = interpolate.interp1d(timestamp_list, RoT_list, kind='nearest')
                    SoG_interpolator = interpolate.interp1d(timestamp_list, SoG_list, kind='linear')
                    CoG_interpolator = interpolate.interp1d(timestamp_list, CoG_list, kind='linear')
                    Th_interpolator = interpolate.interp1d(timestamp_list, Th_list, kind='linear')
                    while (step < timestamp_list[-1]):
                        longitude = long_interpolator(step)
                        latitude = lat_interpolator(step)
                        rate_of_turn = RoT_interpolator(step)
                        speed_over_ground = SoG_interpolator(step)
                        course_over_ground = CoG_interpolator(step)
                        true_heading = Th_interpolator(step)
                        utc_step = datetime.utcfromtimestamp(step).strftime('%Y-%m-%d %H:%M:%S')
                        # day: 1 and night: 0
                        t_day = self._isDay(longitude, latitude, utc_step)
                        # Monday: 0 - Tuesday: 1 - Wednesday: 2, Thursday: 3 - Friday: 4 - Saturday: 5 - Sunday: 6
                        t_week = self._weekDay(datetime.utcfromtimestamp(step).strftime('%Y-%m-%d'))
                        vRoute.states.append(state.State(number_in_route, longitude, latitude, status, rate_of_turn, speed_over_ground, course_over_ground, true_heading, t_day, t_week, draught))
                        number_in_route += 1
                        step += vRoute.TIME_STEP
                    vRoute.t_end = step - vRoute.TIME_STEP
                    ship.routes.append(vRoute)
                    self.__clearLists(timestamp_list, long_list, lat_list, RoT_list, SoG_list, CoG_list, Th_list)

    def __computeMaxMin(self):
        """ computes the minimum and maximum of features to normalize them in a next step
        """
        for ship in self.ships:
            max_length = min_length = ship.length
            max_width = min_width = ship.width
            if (len(ship.ais_states) > 0):
                vState = ship.ais_states[0]
                break

        max_long = min_long = vState.longitude
        max_lat = min_lat = vState.latitude
        max_RoT = min_RoT = vState.rate_of_turn
        max_CoG = min_CoG = vState.course_over_ground
        max_SoG = min_SoG = vState.speed_over_ground
        max_Th = min_Th = vState.true_heading
        max_eta = min_eta = vState.eta

        for ship in self.ships:
            # max min length
            max_length = max(max_length, ship.length)
            min_length = min(min_length, ship.length)
            # max min width
            max_width = max(max_width, ship.width)
            min_width = min(min_width, ship.width)
            for st in ship.ais_states:
                # max min longitude
                max_long = max(max_long, st.longitude)
                min_long = min(min_long, st.longitude)
                # max min latitude
                max_lat = max(max_lat, st.latitude)
                min_lat = min(min_lat, st.latitude)
                # max min rate_of_turn
                max_RoT = max(max_RoT, st.rate_of_turn)
                min_RoT = min(min_RoT, st.rate_of_turn)
                # max min course_over_ground
                max_CoG = max(max_CoG, st.course_over_ground)
                min_CoG = min(min_CoG, st.course_over_ground)
                # max min speed_over_ground
                max_SoG = max(max_SoG, st.speed_over_ground)
                min_SoG = min(min_SoG, st.speed_over_ground)
                # max min true_heading
                max_Th = max(max_Th, st.true_heading)
                min_Th = min(min_Th, st.true_heading)
                # max min eta
                max_eta = max(max_eta, st.eta)
                min_eta = min(min_eta, st.eta)

        return max_long, min_long, max_lat, min_lat, max_RoT, min_RoT, max_CoG, min_CoG, max_SoG, min_SoG, max_Th, min_Th, max_eta, min_eta, max_length, min_length, max_width, min_width

    def __convertShiptype(self, shiptype):
        """ one hot encodes the shiptype
        return: array of zero except for the index of the given shiptype
        """
        shiptypearray = np.zeros(4)
        if (shiptype == self.FISHING):
            shiptypearray[0] = 1
        elif (shiptype in self.PASSENGER):
            shiptypearray[1] = 1
        elif (shiptype in self.CARGO):
            shiptypearray[2] = 1
        elif (shiptype in self.TANKER):
            shiptypearray[3] = 1
        return shiptypearray

    def __convertDestination(self, destination):
        """ one hot encodes the destination
        return: binary array of zeros except for the index of the given destination
        """
        destinationarray = np.zeros(len(self.all_destination))
        destinationarray[self.all_destination[destination]] = 1
        return destinationarray

    def __convertStatus(self, status):
        """ one hot encodes the status
        return: binary array of zeros except for the index of the given status
        """
        statusarray = np.zeros(len(self.all_status))
        statusarray[self.all_status[status]] = 1
        return statusarray

    def __convertWeekday(self, week_day):
        """ one hot encodes the week day
        return: binary array of zeros except for the index of the given week day
        """
        t_weekarray = np.zeros(7)
        if (week_day == 0):
            t_weekarray[0] = 1
        elif (week_day == 1):
            t_weekarray[1] = 1
        elif (week_day == 2):
            t_weekarray[2] = 1
        elif (week_day == 3):
            t_weekarray[3] = 1
        elif (week_day == 4):
            t_weekarray[4] = 1
        elif (week_day == 5):
            t_weekarray[5] = 1
        elif (week_day == 6):
            t_weekarray[6] = 1
        return t_weekarray

    def __normalizeValue(self, value, max, min):
        """ Normalizes a value with its maximum and minimum

        return: a normalized value in [0-1]
        """
        if (max - min != 0):
            return (value - min) / (max - min)
        else:
            return 0.0

    def __setMinMaxScope(self, max_long, min_long, max_lat, min_lat):
        """ Sets the maximum and minimum of longitude and latidude of the available ships to scope
        """
        self.scope.longitude_max = max_long
        self.scope.longitude_min = min_long
        self.scope.latitude_max = max_lat
        self.scope.latitude_min = min_lat

    def normalize(self):
        """ Normalizes all features
        """
        max_long, min_long, max_lat, min_lat, max_RoT, min_RoT, max_CoG, min_CoG, max_SoG, min_SoG, max_Th, min_Th, max_eta, min_eta, max_length, min_length, max_width, min_width = self.__computeMaxMin()

        self.__setMinMaxScope(max_long, min_long, max_lat, min_lat)

        for ship in self.ships:
            ship.shiptypearray = self.__convertShiptype(ship.shiptype)
            ship.normalized_length = self.__normalizeValue(ship.length, max_length, min_length)
            ship.normalized_width = self.__normalizeValue(ship.width, max_width, min_width)
            for vRoute in ship.routes:
                vRoute.destinationarray = self.__convertDestination(vRoute.destination)
                if (vRoute.eta == 0):
                    vRoute.valid_eta = False
                else:
                    vRoute.eta = self.__normalizeValue(vRoute.eta, max_eta, min_eta)

                for st in vRoute.states:
                    st.longitude = self.__normalizeValue(st.longitude, max_long, min_long)
                    st.latitude = self.__normalizeValue(st.latitude, max_lat, min_lat)
                    st.statusarray = self.__convertStatus(st.status)
                    if (st.rate_of_turn == -128):
                        st.valid_rate_of_turn = False
                    st.rate_of_turn = self.__normalizeValue(st.rate_of_turn, max_RoT, min_RoT)
                    st.speed_over_ground = self.__normalizeValue(st.speed_over_ground, max_SoG, min_SoG)
                    if (st.course_over_ground == 360):
                        st.valid_course_over_ground = False
                    st.course_over_ground = self.__normalizeValue(st.course_over_ground, max_CoG, min_CoG)
                    if (st.true_heading == 511):
                        st.valid_true_heading = False
                    st.true_heading = self.__normalizeValue(st.true_heading, max_Th, min_Th)
                    st.t_weekarray = self.__convertWeekday(st.t_week)
                    st.draught = (st.draught / state.draught_max)

    def getScope(self):
        """ Applies all methods to clean and preprocess ships and sets the preprocessed ships to scope

        return: scope with preprocessed ships
        """
        self.filter_by_scope()
        self.filter_by_shiptype()
        self.delete_not_moving_ships()
        self.delete_short_sequences()
        self.filter_by_NaN()
        self.handle_missing_values()
        self.create_routes()
        self.normalize()

        self.scope.ships = self.ships
        return self.scope


def arg_parser():
    """ returns an argument parse, which parses the arugments, which are needed.
    """
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('scope definition')
    group.add_argument("--longitude_min", action='store', type=float, help="the most western longitude to use")
    group.add_argument("--longitude_max", action='store', type=float, help="the most eastern longitude to use")
    group.add_argument("--latitude_min", action='store', type=float, help="the most southern latitude to use")
    group.add_argument("--latitude_max", action='store', type=float, help="the most northern latitude to use")
    group.add_argument("--t_min", action='store', type=int, default=0, help="the start of time horizont (Unix timestamp)")
    group.add_argument("--t_max", action='store', type=int, default=sys.maxsize, help="the end of time horizont (Unix timestamp)")
    input_target = parser.add_mutually_exclusive_group(required=True)
    db_target = input_target.add_argument_group("database credentials")
    db_target.add_argument("--url", action='store', type=str, help="the URL to the database.")
    db_target.add_argument("--db", action='store', type=str, help="the name of the database.")
    db_target.add_argument("--user", action='store', type=str, help="the user name to use for access.")
    db_target.add_argument("--password", action='store', type=str, help="the user password to use for access.")
    input_target.add_argument("--nari-dir", action='store', type=str, help="the directory, which contains the NARI-dataset")
    return parser


def importer_for_csv(directory):
    from data_import.aisimport import AISImporterCSV
    importer = AISImporterCSV(directory + "/nari_static.csv",
                              directory + "/nari_dynamic.csv")
    return importer.import_data()


def importer_for_db(args, user_scope):
    from data_import.aisimport import AISImporterPostgres
    importer = AISImporterPostgres(args.urls, args.db, args.user, args.password)
    return importer.import_data((user_scope.t_min, user_scope.t_max),
                                (user_scope.latitude_min, user_scope.latitude_max),
                                (user_scope.longitude_min, user_scope.longitude_max))


if __name__ == "__main__":
    import argparse
    from persistent.writer import Writer
    import persistent.argument
    from aisdata.scope import Scope

    parser = argparse.ArgumentParser(parents=[persistent.argument.output_path_parser(), arg_parser()])
    args = parser.parse_args()
    user_scope = Scope(args.longitude_min, args.longitude_max, args.latitude_min, args.latitude_max, args.t_min, args.t_max)
    ships = importer_for_csv(args.nari_dir) if args.nari_dir else importer_for_db(args, user_scope)

    prep = Preprocessing(user_scope, ships)
    scope = prep.getScope()
    writer = Writer(args.output)
    writer.write(scope)
