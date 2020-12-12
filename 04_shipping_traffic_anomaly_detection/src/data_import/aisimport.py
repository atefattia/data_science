#!/usr/bin/env python3

"""Organizes data imports from external sources"""
import csv
import psycopg2
import sys
import traceback
from datetime import datetime

from aisdata import ship
from aisdata import aisstate

def read_float(row, field):
    """ read float from fiel of rowd

    return NaN if field contains not a number"""
    try:
        return float(row[field])
    except Exception:
        return float('NaN')

def read_int(row, field, default=0):
    """ read int from fiel of rowd

    return default if field contains not a number"""
    try:
        return float(row[field])
    except Exception:
        return default

def read_string(row, field):
    """ read string from field of row"""
    try:
        return row[field].strip()
    except Exception:
        return ""

def read_sourcemmsi(row):
    """ read sourcemmsi from field of row

    Raise exception if sourcemmsi not found."""
    try:
        return row["sourcemmsi"].strip()
    except Exception:
        raise Exception("No valid Sourcemmsi found!")

class _ImportedShip(ship.Ship):

    """Class for managing the data import into the Ship class."""

    def __init__(self, sourcemmsi, shipname, shiptype, length, width):
        ship.Ship.__init__(self, sourcemmsi, shipname, shiptype, length, width)
        self.ais_static_states = []

    def merge_static_into_dynamic_states(self):
        """Write the temporary list of static states into the dynamic ones."""
        key = lambda a: a.timestamp
        self.ais_static_states.sort(key=key)
        self.ais_states.sort(key=key)
        static = self.ais_static_states[0]
        static_iterator = iter(self.ais_static_states)
        next_static = next(static_iterator)
        for dynamic in self.ais_states:
            while dynamic.timestamp >= next_static.timestamp:
                try:
                    static = next_static
                    next_static = next(static_iterator)
                except StopIteration:
                    break
            dynamic.destination = static.destination
            dynamic.eta = static.eta
            dynamic.draught = static.draught



class AISImporter:

    """Abstract class for importing AIS data from different external sources"""

    def __init__(self):
        pass

    def import_data(self, time_constraints=None, lat_constraints=None, lon_constraints=None):
        """Imports the data returns a list of ships based on the given AIS
        data."""
        ships = {}
        self._import_from_source(ships, time_constraints, lat_constraints, lon_constraints)
        for _, s in ships.items():
            s.merge_static_into_dynamic_states()
        return list(ships.values())

    def _import_from_source(self, ships, time_constraints, lat_constraints, lon_constraints):
        """Imports the data from the given source, and writes them to the given list.

        This method is abstract and has to be implemented by child class.

        Parameters:
        ships: list of ships to save the data at"""
        pass

class AISImporterCSV(AISImporter):

    """Importing AIS data from external csv files"""

    def __init__(self, static_file, dynamic_file):
        """Creates object for csv import from given file paths.

        Parameters:
        static_file: path to csv file with static ais data
        dynamic_file: path to csv file with dynamic ais data
        """
        AISImporter.__init__(self)
        self.static_file = static_file
        self.dynamic_file = dynamic_file

    def _import_from_source(self, ships, time_constraints, lat_constraints, lon_constraints):
        """Imports the data from the given csv files.

        Parameters:
        ships: list of ships to save the data at
        All other parameters are ignored yet."""
        self._import_static(ships)
        self._import_dynamic(ships)

    def _import_static(self, ships):
        """Imports the static ais data.

        Parameters:
        ships: list of ships to save the data at"""
        with open(self.static_file) as inputfile:
            reader = csv.DictReader(inputfile, delimiter=',')
            for row in reader:
                sourcemmsi = read_sourcemmsi(row)
                shipname = read_string(row, "shipname")
                shiptype = read_int(row, "shiptype")
                try:
                    tobow = float(row["tobow"])
                    tostern = float(row["tostern"])
                    length = tobow + tostern
                except Exception:
                    length = float('NaN')
                try:
                    tostarboard = float(row["tostarboard"])
                    toport = float(row["toport"])
                    width = tostarboard + toport
                except Exception:
                    width = float('NaN')
                eta = read_string(row, "eta")
                try:
                    eta_time = datetime.strptime(eta, '%d-%m %H:%M')
                    eta_time.replace(year=2000)
                    eta_epoch = int(eta_time.timestamp())
                except ValueError:
                    eta_epoch = 0
                draught = read_float(row, "draught")
                destination = read_string(row, "destination")
                t = read_int(row, "t")
                if sourcemmsi not in ships:
                    ships[sourcemmsi] =\
                            _ImportedShip(sourcemmsi, shipname, shiptype, length, width)
                ships[sourcemmsi].ais_static_states.append(\
                        aisstate.AISState(0, 0, 0, 0, 0, 0, 0, t, destination, eta_epoch, draught))

    def _import_dynamic(self, ships):
        """Imports the dynamic ais data.

        Parameters:
        ships: list of ships to save the data at"""
        with open(self.dynamic_file) as inputfile:
            reader = csv.DictReader(inputfile, delimiter=',')
            for row in reader:
                sourcemmsi = read_sourcemmsi(row)
                navigationalstatus = read_int(row, "navigationalstatus", 15)
                rateofturn = read_float(row, "rateofturn")
                speedoverground = read_float(row, "speedoverground")
                courseoverground = read_float(row, "courseoverground")
                trueheading = read_float(row, "trueheading")
                lon = read_float(row, "lon")
                lat = read_float(row, "lat")
                t = read_int(row, "t")
                if sourcemmsi in ships:
                    ships[sourcemmsi].ais_states.append(\
                        aisstate.AISState(lon, lat, navigationalstatus, rateofturn, \
                        speedoverground, courseoverground, trueheading, t, "", 0, 0))

class AISImporterPostgres(AISImporter):

    """Import ais data from IOSB Postgresql database."""

    def __init__(self, host, database, user, password):
        """Creates object for database import.

        Parameters:
        host: hostname of database server
        database: name of database
        user: username for login at database
        password: password for user
        """
        AISImporter.__init__(self)
        self.connect_str = "dbname='"+database+"' user='"+user+"' host='"+host+"' " + \
                          "password='"+password+"'"
        # test connection
        try:
            conn = psycopg2.connect(self.connect_str)
            conn.close()
        except Exception as e:
            print("Can not connect to database!")
            print(e)

    def _import_from_source(self, ships, time_constraints, lat_constraints, lon_constraints):
        """Imports the data from the given source, and writes them to the given list.

        This method is abstract and has to be implemented by child class.

        Parameters:
        ships: list of ships to save the data at"""
        query = ( 'SELECT name.mmsi, EXTRACT(EPOCH FROM name.timedate), name.speed, name.course, name.heading, name.lon, name.lat, ' +
                  'name.shipname, name.shiptype, name.to_bow, name.to_stern, name.to_starboard, name.to_port, name.draught, dest.destination, dest.eta, MIN(name.timediff)' +
                  'FROM (' +
                  '    SELECT p.mmsi, p.timedate, p.speed, p.course, p.heading, ST_X(p.position) as lon, ST_Y(p.position) as lat,' +
                  '        v.shipname, v.shiptype, v.to_bow, v.to_stern, v.to_starboard, v.to_port, v.draught,' +
                  '        MIN(p.timedate - d.timedate) AS timediff' +
                  '    FROM public.vessels v, public.positions p ' +
                  '    INNER JOIN public.destinations d ON p.mmsi = d.mmsi ' +
                  '    WHERE p.mmsi = v.mmsi ' +
                  '        and p.timedate between (to_timestamp(%s) AT TIME ZONE \'UTC\') ' +
                  '            and (to_timestamp(%s) AT TIME ZONE \'UTC\') ' +
                  '        and  position && ST_MakeEnvelope(%s, %s, %s, %s) ' +
                  '        and p.timedate > d.timedate' +
                  '    GROUP BY p.mmsi, p.timedate, p.speed, p.course, p.heading, lon, lat,' +
                  '        v.shipname, v.shiptype, v.to_bow, v.to_stern, v.to_starboard, v.to_port, v.draught' +
                  '    ORDER BY p.mmsi, p.timedate, timediff' +
                  '    ) ' +
                  'AS name, public.destinations dest WHERE name.mmsi = dest.mmsi and name.timedate = dest.timedate + name.timediff' +
                  '    GROUP BY name.mmsi, name.timedate, name.speed, name.course, name.heading, name.lon, name.lat,' +
                  '        name.shipname, name.shiptype, name.to_bow, name.to_stern, name.to_starboard, name.to_port, name.draught, dest.destination, dest.eta ' +
                  '    ORDER BY name.mmsi, name.timedate')
        try:
            conn = psycopg2.connect(self.connect_str)
            cursor = conn.cursor()
            #print("Query is:")
            #print(query % (str(time_constraints[0]) , str(time_constraints[1]) ,
            #    lon_constraints[0], lat_constraints[0], lon_constraints[1], lat_constraints[1]))
            cursor.execute(query, (int(time_constraints[0]), int(time_constraints[1]),
                                   lon_constraints[0], lat_constraints[0],
                                   lon_constraints[1], lat_constraints[1]))
            for row in cursor:
                try:
                    #print(row)
                    self.add_to_ships(ships, row)
                except Exception as e:
                    print("Error while parsing from database:" + str(row))
                    print(e)
                    print(traceback.format_exc())
            cursor.close()
            conn.close()
        except Exception as e:
            print("Can not connect to database!")
            print(e)
            print(traceback.format_exc())

    def add_to_ships(self, ships, row):
        """ creates new entry to ships from sql query result. """
        sourcemmsi = row[0]
        t = int(row[1])
        speedoverground = float(row[2])
        courseoverground = float(row[3])
        trueheading = float(row[4])
        lon = float(row[5])
        lat = float(row[6])
        shipname = row[7].strip()
        shiptype = int(row[8])
        try:
            tobow = float(row[9])
            tostern = float(row[10])
            length = tobow + tostern
        except Exception:
            length = float('NaN')
        try:
            tostarboard = float(row[11])
            toport = float(row[12])
            width = tostarboard + toport
        except Exception:
            width = float('NaN')
        draught = float(row[13])
        destination = row[14]
        eta = row[15]
        try:
            eta_time = datetime.strptime(eta, '%d-%mT%H:%MZ')
            eta_time.replace(year=2000)
            eta_epoch = int(eta_time.timestamp())
        except ValueError:
            eta_epoch = 0

        if sourcemmsi not in ships:
            ships[sourcemmsi] =\
                    _ImportedShip(sourcemmsi, shipname, shiptype, length, width)
        ships[sourcemmsi].ais_static_states.append(\
                aisstate.AISState(lon, lat, float('NaN'), float('NaN'), \
                speedoverground, courseoverground, trueheading, t, destination, eta_epoch, draught))


def arg_parser():
    """ returns an argument parse, which parses the arugments, which are needed.
    """
    parser = argparse.ArgumentParser(add_help=False)
    input_target = parser.add_mutually_exclusive_group(required=True)
    db_target = input_target.add_argument_group("database credentials")
    group = db_target.add_argument_group('scope definition')
    group.add_argument("--longitude_min", action='store', type=float, help="the most western longitude to use")
    group.add_argument("--longitude_max", action='store', type=float, help="the most eastern longitude to use")
    group.add_argument("--latitude_min", action='store', type=float, help="the most southern latitude to use")
    group.add_argument("--latitude_max", action='store', type=float, help="the most northern latitude to use")
    group.add_argument("--t_min", action='store', type=int, default=0, help="the start of time horizont (Unix timestamp)")
    group.add_argument("--t_max", action='store', type=int, default=sys.maxsize, help="the end of time horizont (Unix timestamp)")
    db_target.add_argument("--url", action='store', type=str, help="the URL to the database.")
    db_target.add_argument("--db", action='store', type=str, help="the name of the database.")
    db_target.add_argument("--user", action='store', type=str, help="the user name to use for access.")
    db_target.add_argument("--password", action='store', type=str, help="the user password to use for access.")
    input_target.add_argument("--nari-dir", action='store', type=str, help="the directory, which contains the NARI-dataset")
    return parser

def importer_for_csv(directory):
    importer = AISImporterCSV(directory + "nari_static.csv",
                              directory + "nari_dynamic.csv")
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
    user_scope.ships = importer_for_csv(args.nari_dir) if args.nari_dir else importer_for_db(args, user_scope)
    writer = Writer(args.output)
    writer.write(user_scope)
