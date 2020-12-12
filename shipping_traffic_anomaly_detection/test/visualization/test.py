import os

from data_import import aisimport
from visualization import qt_frame
from data_preprocessing import preprocessing
from aisdata import scope

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

    # Create PyQt5-Application
    vis = qt_frame.QtFrame(
        routes_line_color=(0.6, 0.2, 0.8),
        routes_marker_color=(0, 0, 1.0),
        predictions_line_color=(1.0, 0, 0),
        predictions_marker_color=(0, 0, 1.0),
        marker_size=0.75,
        line_thickness=0.5,
        mark_every=5)

    # Create content for plot
    vis.set_scope(s)
    vis.draw_map()
    vis.draw_routes()
    vis.draw_predictions()
    vis.show_dialog()
