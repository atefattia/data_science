import os

from data_import import aisimport
from visualization import qt_frame
from data_preprocessing import preprocessing
from aisdata import scope

import random

def set_last_n_as_predicted(scope,  n):
    """ Changes last n route positions as prediction and returns the resulting scope
    Parameters:
    scope: the scope to change
    n: number of positions to change
    """
    for i, ship in enumerate(scope.ships):
        for j, route in reversed(list(enumerate(ship.routes))):
            num_routes = len(ship.routes)
            if j == num_routes - 1:
                predict = []
                anomalies = []
                m = n
                for k, state in reversed(list(enumerate(route.states))):
                    if m > 0:
                        predict.append(state)
                        anomalies.append(random.uniform(0, 1))
                        route.states.pop()
                        m = m - 1
                if len(route.states) == 0:
                    ship.routes.pop()
                route.predicted_states = predict[::-1]
                route.anomalies = anomalies[::-1]
    return scope

if __name__ == '__main__':
    # Import data from nari data set
    directory = os.path.dirname(os.path.abspath(__file__)) + "/../examples/"
    importer = aisimport.AISImporterCSV(directory + "nari_static.csv",
                                        directory + "nari_dynamic.csv")
    ships = importer.import_data()

    # Data preprocessing
    test_scope = scope.Scope(-7, -2, 40, 50, 0, 1600000000)
    preprocessing = preprocessing.Preprocessing(test_scope, ships)

    s = preprocessing.getScope()

    # Use last 50 route positions as predictions
    s = set_last_n_as_predicted(s, 30)

    # Create PyQt5-Application
    vis = qt_frame.QtFrame(
        routes_line_color=(0.6, 0.2, 0.8),
        routes_marker_color=(0, 0, 1.0),
        predictions_line_color=(1, 0.4, 0),
        predictions_marker_color=(0.8, 0, 0),
        marker_size=0.75,
        line_thickness=0.5,
        mark_every=5)

    # Create content for plot
    vis.set_scope(s)
    vis.draw_map()
    vis.draw_routes()
    vis.draw_predictions()
    vis.show_dialog()
