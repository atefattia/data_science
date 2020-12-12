import os
import unittest

from data_import import aisimport


class TestAISImport(unittest.TestCase):

    """Tests the import of external AIS data."""

    def test_csv_import(self):
        """Test the import of external AIS data from csv files."""
        directory = os.path.dirname(os.path.abspath(__file__)) + "/examples/"
        importer = aisimport.AISImporterCSV(directory + "nari_static.csv",
                                            directory + "nari_dynamic.csv")
        ships = importer.import_data()
        self.assertEqual(len(ships), 15)
        number_states = {'304091000': 1, '245257000': 29, '228037600': 259,
                         '227705102': 249, '227002330': 0, '227003050': 0, '227443000': 92,
                         '234056000': 0, '311153000': 0, '311018500': 0, '228931000': 21,
                         '228131600': 350, '228064900': 4, '37100300': 7, '227631770': 3}
        number_static_states = {'304091000': 22, '245257000': 17, '228037600': 16,
                                '227705102': 8, '227002330': 21, '227003050': 15, '227443000': 3,
                                '234056000': 9, '311153000': 13, '311018500': 21, '228931000': 3,
                                '228131600': 13, '228064900': 2, '37100300': 3, '227631770': 2}
        for ship in ships:
            self.assertEqual(len(ship.ais_states), number_states[ship.sourcemmsi],
                             msg="sourcemmsi is " + ship.sourcemmsi)
            self.assertEqual(len(ship.ais_static_states), number_static_states[ship.sourcemmsi],
                             msg="sourcemmsi is " + ship.sourcemmsi)
        for ship in ships:
            for state in ship.ais_states:
                if state.timestamp > 1443653056:
                    self.assertEqual(state.destination, "TESTINATION",
                                     msg="shipname is " + ship.shipname +
                                     ", timestamp is " + str(state.timestamp))
                else:
                    self.assertNotEqual(state.destination, "TESTINATION",
                                        msg="shipname is " + ship.shipname +
                                        ", timestamp is " + str(state.timestamp))


if __name__ == '__main__':
    unittest.main()
