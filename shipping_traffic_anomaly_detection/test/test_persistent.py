import os
import unittest

from aisdata.scope import Scope
from persistent.reader import Reader
from persistent.writer import Writer

def generate_test_scope():
    #TODO implement
    return Scope(-4.9, -4.7, 48, 49, 0, 1500000000)

class TestPersistent(unittest.TestCase):

    """Tests reading and writing scope data to perisitent files."""

    def setUp(self):
        self.test_dir= os.path.dirname(os.path.abspath(__file__)) + "/examples/persistent/"

    def test_reader(self):
        """Test reading data from csv files."""
        reader = Reader(self.test_dir)
        scope = reader.read()
        self.assertEqual(6, len(scope.ships))
        number_ais_states = {'245257000': 29, '228037600': 259, '227705102': 249,
                             '227443000': 92, '228931000': 21, '228131600': 350}
        for ship in scope.ships:
            self.assertEqual(len(ship.ais_states), number_ais_states[ship.sourcemmsi],
                             msg="sourcemmsi is " + ship.sourcemmsi)
        for ship in scope.ships:
            for state in ship.ais_states:
                if state.timestamp > 1443653056:
                    self.assertEqual(state.destination, "TESTINATION",
                                     msg="shipname is " + ship.shipname +
                                     ", timestamp is " + str(state.timestamp))
                else:
                    self.assertNotEqual(state.destination, "TESTINATION",
                                        msg="shipname is " + ship.shipname +
                                        ", timestamp is " + str(state.timestamp))
        number_routes = {'245257000': 2, '228037600': 2, '227705102': 2,
                         '227443000': 1, '228931000': 2, '228131600': 2}
        for ship in scope.ships:
            self.assertEqual(len(ship.routes), number_routes[ship.sourcemmsi],
                             msg="sourcemmsi is " + ship.sourcemmsi)
        for ship in scope.ships:
            for route in ship.routes:
                if route.t_begin > 1443653056:
                    self.assertEqual(route.destination, "TESTINATION",
                                     msg="shipname is " + ship.shipname +
                                     ", t_begin is " + str(route.t_begin))
                else:
                    self.assertNotEqual(route.destination, "TESTINATION",
                                        msg="shipname is " + ship.shipname +
                                        ", t_begin is " + str(route.t_begin))

    #def test_writer(self):
    #    """Test writing data to csv files."""
    #    scope = generate_test_scope()
    #    writer = Writer("")
    #    writer.write(scope)

    #def test_combination(self):
    #    write = generate_test_scope()
    #    writer = Writer("")
    #    writer.write(write)
    #    reader = Reader(self.test_dir)
    #    read = reader.read()
    #    self.assertEqual(len(write.ships), len(read.ships))

if __name__ == '__main__':
    unittest.main()
