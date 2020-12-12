import csv

class Writer:

    """Writes content of a scope to the given directory."""

    def __init__(self, directory):
        self.directory = directory

    def write(self, scope):
        """Writes the given scope to files."""
        self._write_scope(scope)
        self._write_ships(scope.ships)
        self._write_ais_states(scope.ships)
        self._write_routes(scope.ships)
        self._write_states("states.csv", scope.ships, lambda route: route.states)
        self._write_states("predicted_states.csv", scope.ships, lambda route: route.predicted_states)
        self._write_anomalies(scope.ships)

    def _write_scope(self, scope):
        with open(self.directory + '/scope.csv', 'w') as csvfile:
            fieldnames = ['lon_min', 'lon_max', 'lat_min', 'lat_max','user_lon_min', 'user_lon_max', 'user_lat_min', 'user_lat_max', 't_min', 't_max']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            row = {'lon_min':scope.longitude_min, 'lon_max':scope.longitude_max,
                   'lat_min':scope.latitude_min, 'lat_max':scope.latitude_max,
                   'user_lon_min':scope.longitude_min_user, 'user_lon_max':scope.longitude_max_user,
                   'user_lat_min':scope.latitude_min_user, 'user_lat_max':scope.latitude_max_user,
                   't_min':int(scope.t_min), 't_max':int(scope.t_max)}
            writer.writerow(row)

    def _write_ships(self, ships):
        with open(self.directory + '/ships.csv', 'w') as csvfile:
            fieldnames = ['sourcemmsi', 'shipname', 'shiptype', 'length', 'width']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s in ships:
                row = {'sourcemmsi':s.sourcemmsi, 'shipname':s.shipname,
                       'shiptype':int(s.shiptype), 'length':s.length, 'width':s.width}
                writer.writerow(row)

    def _write_ais_states(self, ships):
        with open(self.directory + '/ais_states.csv', 'w') as csvfile:
            fieldnames = ['sourcemmsi', 'lon', 'lat', 'navigationalstatus', 'rateofturn',
                          'speedoverground', 'courseoverground', 'trueheading',
                          'timestamp', 'destination', 'eta', 'draught']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s in ships:
                for a in s.ais_states:
                    row = {'sourcemmsi':s.sourcemmsi,
                           'lon':a.longitude, 'lat':a.latitude,
                           'navigationalstatus':int(a.status), 'rateofturn':a.rate_of_turn,
                           'speedoverground':a.speed_over_ground,
                           'courseoverground':a.course_over_ground,
                           'trueheading':a.true_heading, 'timestamp':a.timestamp,
                           'destination':a.destination, 'eta':int(a.eta),
                           'draught':a.draught}
                    writer.writerow(row)

    def _write_routes(self, ships):
        with open(self.directory + '/routes.csv', 'w') as csvfile:
            fieldnames = ['sourcemmsi', 'destination', 'eta', 't_begin', 't_end']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s in ships:
                for r in s.routes:
                    row = {'sourcemmsi':s.sourcemmsi, 'destination':r.destination,
                           'eta':int(r.eta), 't_begin':int(r.t_begin), 't_end':int(r.t_end)}
                    writer.writerow(row)

    def _write_states(self, file, ships, select_state):
        with open(self.directory + '/' + file, 'w') as csvfile:
            fieldnames = ['sourcemmsi', 't_begin', 'number_in_route', 'lon', 'lat', 'navigationalstatus', 'rateofturn',
                          'speedoverground', 'courseoverground', 'trueheading',
                          't_day', 't_week', 'draught']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s in ships:
                for r in s.routes:
                    nr = 0
                    for st in select_state(r):
                        row = {'sourcemmsi':s.sourcemmsi, 't_begin':int(r.t_begin),
                               'number_in_route':nr,
                               'lon':st.longitude, 'lat':st.latitude,
                               'navigationalstatus':int(st.status), 'rateofturn':st.rate_of_turn,
                               'speedoverground':st.speed_over_ground,
                               'courseoverground':st.course_over_ground,
                               'trueheading':st.true_heading, 't_day':st.t_day, 't_week':st.t_week,
                               'draught':st.draught}
                        writer.writerow(row)
                        nr += 1

    def _write_anomalies(self, ships):
        with open(self.directory + '/anomalies.csv', 'w') as csvfile:
            fieldnames = ['sourcemmsi', 't_begin', 'number_in_route', 'error', 'anomaly']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s in ships:
                for r in s.routes:
                    nr = 0
                    for a in r.anomalies:
                        row = {'sourcemmsi':s.sourcemmsi, 't_begin':int(r.t_begin),
                               'number_in_route':nr, 'error':a.error, 'anomaly':a.anomaly}
                        writer.writerow(row)
                        nr += 1
