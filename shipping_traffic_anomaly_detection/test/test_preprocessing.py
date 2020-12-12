import os
import unittest
import math

from data_import import aisimport
from aisdata import scope
from data_preprocessing import preprocessing


class TestPreprocessing(unittest.TestCase):
    """ Tests the preprocessing class """

    def setUp(self):
        """ import the ships to use them in test methods
        """
        directory = os.path.dirname(os.path.abspath(__file__)) + "/examples/"
        importer = aisimport.AISImporterCSV(directory + "nari_static.csv",
                                            directory + "nari_dynamic.csv")
        self.test_ships = importer.import_data()

    def test_filter_by_scope(self):
        """ Tests if the method filter_by_scope works well
        """
        test_scope = scope.Scope(-4.9, -4.70, 48, 49, 0, 1500000000)
        test_preprocessing = preprocessing.Preprocessing(test_scope, self.test_ships)
        test_preprocessing.filter_by_scope()
        number_states = {'227443000': 10, '228131600': 255}
        self.assertEqual(len(self.test_ships), 2)
        for ship in self.test_ships:
            self.assertEqual(len(ship.ais_states), number_states[ship.sourcemmsi],
                             msg="sourcemmsi is " + ship.sourcemmsi)

    def test_filter_by_type(self):
        """ Tests if the method filter_by_shiptype works well
        """
        test_scope = scope.Scope(-4.9, -4.70, 48, 49, 0, 1500000000)
        test_preprocessing = preprocessing.Preprocessing(test_scope, self.test_ships)

        self.assertEqual(len(self.test_ships), 15)
        test_preprocessing.filter_by_shiptype()
        self.assertEqual(len(self.test_ships), 13)

        for ship in self.test_ships:
            self.assertTrue((ship.shiptype == 30) or (ship.shiptype in range(60, 69)) or (ship.shiptype in range(70, 79)) or (ship.shiptype in range(80, 89)))

    def test_delete_short_sequences(self):
        """ Tests if the method delete_short_sequences works well
        """
        test_scope = scope.Scope(-4.9, -4.70, 48, 49, 0, 1500000000)
        test_preprocessing = preprocessing.Preprocessing(test_scope, self.test_ships)

        self.assertEqual(len(self.test_ships), 15)
        test_preprocessing.delete_short_sequences()
        self.assertEqual(len(self.test_ships), 7)

        number_states = {'245257000': 29, '228037600': 259, '227705102': 249,
                         '227443000': 92, '228931000': 21, '228131600': 350, '37100300': 7}
        for ship in self.test_ships:
            self.assertEqual(len(ship.ais_states), number_states[ship.sourcemmsi],
                             msg="sourcemmsi is " + ship.sourcemmsi)

    def test_delete_not_moving_ships(self):
        """ Tests if the method delete_not_moving_ships works well
        """
        test_scope = scope.Scope(-4.9, -4.70, 48, 49, 0, 1500000000)
        test_preprocessing = preprocessing.Preprocessing(test_scope, self.test_ships)

        self.assertEqual(len(self.test_ships), 15)
        test_preprocessing.delete_not_moving_ships()
        self.assertEqual(len(self.test_ships), 4)

        for ship in self.test_ships:
            for state in ship.ais_states:
                self.assertTrue(state.speed_over_ground > 0.2)

    def test_filter_by_NaN(self):
        """ test if the method filter_by_NaN works well
        """
        test_scope = scope.Scope(-4.9, -4.70, 48, 49, 0, 1500000000)
        test_preprocessing = preprocessing.Preprocessing(test_scope, self.test_ships)

        self.assertEqual(len(self.test_ships), 15)
        test_preprocessing.filter_by_NaN()
        self.assertEqual(len(self.test_ships), 7)

        number_states = {'245257000': 29, '228037600': 259, '304091000': 1,
                         '227705102': 249, '227443000': 92, '228931000': 21,
                         '228131600': 350, '228064900': 4, '37100300': 7}

        for ship in self.test_ships:
            self.assertEqual(len(ship.ais_states), number_states[ship.sourcemmsi],
                             msg="sourcemmsi is " + ship.sourcemmsi)

    def test_handle_missing_values(self):
        """ test if the method handle_missing_values works well
        """
        test_scope = scope.Scope(-4.9, -4.70, 48, 49, 0, 1500000000)
        test_preprocessing = preprocessing.Preprocessing(test_scope, self.test_ships)

        test_preprocessing.handle_missing_values()

        for ship in self.test_ships:
            self.assertFalse(ship.shipname == "")
            self.assertFalse(math.isnan(ship.length))
            self.assertFalse(math.isnan(ship.width))
            for state in ship.ais_states:
                self.assertFalse(math.isnan(state.rate_of_turn))
                self.assertFalse(math.isnan(state.draught))
                self.assertFalse(state.destination == "")

    def test_create_routes(self):
        """
        """
        test_scope = scope.Scope(-4.9, -4.70, 48, 49, 0, 1500000000)
        test_preprocessing = preprocessing.Preprocessing(test_scope, self.test_ships)

        test_preprocessing.filter_by_shiptype()
        # test_preprocessing.delete_short_sequences()
        # test_preprocessing.delete_not_moving_ships()
        test_preprocessing.filter_by_NaN()
        test_preprocessing.handle_missing_values()
        test_preprocessing.create_routes()

    def test_normalize(self):
        test_scope = scope.Scope(-4.9, -4.70, 48, 49, 0, 1500000000)
        test_preprocessing = preprocessing.Preprocessing(test_scope, self.test_ships)

        test_preprocessing.filter_by_shiptype()
        test_preprocessing.delete_short_sequences()
        test_preprocessing.handle_missing_values()
        test_preprocessing.create_routes()
        test_preprocessing.normalize()

        for ship in self.test_ships:
            self.assertTrue(ship.normalized_length >= 0.0 and ship.normalized_length <= 1.0)
            self.assertTrue(ship.normalized_width >= 0.0 and ship.normalized_width <= 1.0)
            for route in ship.routes:
                self.assertTrue(route.eta >= 0.0 and route.eta <= 1.0)
                for state in route.states:
                    self.assertTrue(state.longitude >= 0.0 and state.longitude <= 1.0)
                    self.assertTrue(state.latitude >= 0.0 and state.latitude <= 1.0)
                    self.assertTrue(state.rate_of_turn >= 0.0 and state.rate_of_turn <= 1.0)
                    self.assertTrue(state.speed_over_ground >= 0.0 and state.speed_over_ground <= 1.0)
                    self.assertTrue(state.course_over_ground >= 0.0 and state.course_over_ground <= 1.0)
                    self.assertTrue(state.true_heading >= 0.0 and state.true_heading <= 1.0)
                    self.assertTrue(state.draught >= 0.0 and state.draught <= 1.0)

    def test_getScope(self):
        test_scope = scope.Scope(-7.0, -2.0, 40, 50, 0, 1600000000)
        test_preprocessing = preprocessing.Preprocessing(test_scope, self.test_ships)

        vScope = test_preprocessing.getScope()
        self.assertEqual(len(vScope.ships), 4)

    def test_isDay(self):
        """ test if the method _isDay() works well
        """
        test_scope = scope.Scope(-7.0, -2.0, 40, 50, 0, 1600000000)
        test_preprocessing = preprocessing.Preprocessing(test_scope, self.test_ships)

        self.assertEqual(test_preprocessing._isDay(-4.77353, 48.015366, "2015-09-30 10:25:24"), 1)
        self.assertEqual(test_preprocessing._isDay(-4.77353, 48.015366, "2015-09-30 12:25:24"), 1)
        self.assertEqual(test_preprocessing._isDay(-4.77353, 48.015366, "2015-09-30 14:25:24"), 1)
        self.assertEqual(test_preprocessing._isDay(-4.77353, 48.015366, "2015-09-30 16:25:24"), 1)
        self.assertEqual(test_preprocessing._isDay(-4.77353, 48.015366, "2015-09-30 18:25:24"), 1)
        self.assertEqual(test_preprocessing._isDay(-4.77353, 48.015366, "2015-09-30 19:25:24"), 0)
        self.assertEqual(test_preprocessing._isDay(-4.77353, 48.015366, "2015-09-30 21:25:24"), 0)
        self.assertEqual(test_preprocessing._isDay(-4.77353, 48.015366, "2015-09-30 23:25:24"), 0)
        self.assertEqual(test_preprocessing._isDay(-4.77353, 48.015366, "2015-09-30 00:25:24"), 0)
        self.assertEqual(test_preprocessing._isDay(-4.77353, 48.015366, "2015-09-30 02:25:24"), 0)

    def test_weekDay(self):
        """ test if the method _weekDay() works well
        """
        test_scope = scope.Scope(-7.0, -2.0, 40, 50, 0, 1600000000)
        test_preprocessing = preprocessing.Preprocessing(test_scope, self.test_ships)

        # Monday: 0
        self.assertEqual(test_preprocessing._weekDay("2018-12-03"), 0)
        # Tuesday: 1
        self.assertEqual(test_preprocessing._weekDay("2018-12-04"), 1)
        # Wednesday: 2
        self.assertEqual(test_preprocessing._weekDay("2018-12-05"), 2)
        # Thursday: 3
        self.assertEqual(test_preprocessing._weekDay("2018-12-06"), 3)
        # Friday: 4
        self.assertEqual(test_preprocessing._weekDay("2018-12-07"), 4)
        # Saturday: 5
        self.assertEqual(test_preprocessing._weekDay("2018-12-08"), 5)
        # Sunday: 6
        self.assertEqual(test_preprocessing._weekDay("2018-12-09"), 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
