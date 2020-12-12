
""" PyQt5-frame for visualization """

import sys
import argparse
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QAction,\
    QPushButton, QCheckBox, QColorDialog, QInputDialog, QSlider, QLabel
from PyQt5.QtCore import Qt

from visualization.vis import Visualization

class QtFrame(QWidget):
    def __init__(self,
                 routes_line_color=(0.6, 0.2, 0.8),
                 routes_marker_color=(0, 0, 1.0),
                 predictions_line_color=(1.0, 0.4, 0),
                 predictions_marker_color=(0, 0, 1.0),
                 marker_size=0.75,
                 line_thickness=0.5,
                 mark_every=5,
                 anomaly_threshold=0.5):
        """ Constructor of QtFrame
        Parameters:
        routes_line_color: the color for route lines
        routes_marker_color: the color for route markers
        predictions_line_color: the color for prediction lines
        predictions_marker_color: the color for prediction markers
        marker_size: the size of markers
        line_thickness: the width of lines
        mark_every: the number to show only every n-th marker
        """
        # Create a PyQt5-Application
        self.app = QApplication(sys.argv)
        QWidget.__init__(self)

        self.line_thickness = line_thickness
        self.marker_size = marker_size
        self.mark_every = mark_every
        self.anomaly_threshold = anomaly_threshold

        # Create the plot and a matplotlib NavigationToolbar
        self.plot_layout = QVBoxLayout(self)
        self.vis = Visualization(routes_line_color=routes_line_color,
                                 routes_marker_color=routes_marker_color,
                                 predictions_line_color=predictions_line_color,
                                 predictions_marker_color=predictions_marker_color,
                                 marker_size=marker_size,
                                 line_thickness=line_thickness,
                                 mark_every=mark_every,
                                 anomaly_threshold=anomaly_threshold)
        self.navi_toolbar = NavigationToolbar(self.vis, self)

        # Remove not used buttons inside the navigation toolbar
        actions = self.navi_toolbar.findChildren(QAction)
        for a in actions:
            if a.text() == "Customize" or a.text() == "Subplots":
                self.navi_toolbar.removeAction(a)

        # Create custom buttons and checkbox to interact with the plot
        self.settings_row = QHBoxLayout()
        self.settings_row_2 = QHBoxLayout()
        self.settings_row_2.setContentsMargins(0, 0, 0, 10)
        self.button_zoom = QPushButton("Zoom to calculated scope")
        self.button_routes_line_color = QPushButton("Set routes line color")
        self.button_routes_marker_color = QPushButton("Set routes marker color")
        self.button_prediction_line_color = QPushButton("Set prediction line color")
        self.button_prediction_marker_color = QPushButton("Set prediction marker color")
        self.button_line_width = QPushButton("Set line width")
        self.button_marker_size = QPushButton("Set marker size")
        self.button_mark_every = QPushButton("Set every n-th marker")
        self.checkbox_anomalies = QCheckBox("Show Anomalies")
        self.label_anomaly_threshold = QLabel("Anomaly threshold: %.4f" % round(self.anomaly_threshold, 5))
        self.label_anomaly_threshold.setFixedWidth(200)
        self.slider_anomaly_threshold = QSlider(Qt.Horizontal)
        self.slider_anomaly_threshold.setFixedWidth(200)
        self.slider_anomaly_threshold.setTickPosition(QSlider.NoTicks)
        self.slider_anomaly_threshold.setRange(0, 10000)
        self.slider_anomaly_threshold.setValue(self.anomaly_threshold * 10000)

        # Add layouts and widgets
        self.plot_layout.addWidget(self.vis)
        self.plot_layout.addWidget(self.navi_toolbar)
        self.plot_layout.addLayout(self.settings_row)
        self.plot_layout.addLayout(self.settings_row_2)
        self.settings_row.addWidget(self.button_zoom)
        self.settings_row.addWidget(self.button_routes_line_color)
        self.settings_row.addWidget(self.button_routes_marker_color)
        self.settings_row.addWidget(self.button_prediction_line_color)
        self.settings_row.addWidget(self.button_prediction_marker_color)
        self.settings_row.addWidget(self.button_line_width)
        self.settings_row.addWidget(self.button_marker_size)
        self.settings_row.addWidget(self.button_mark_every)
        self.settings_row.addStretch(1)
        self.settings_row_2.addWidget(self.checkbox_anomalies)
        self.settings_row_2.addWidget(self.label_anomaly_threshold)
        self.settings_row_2.addWidget(self.slider_anomaly_threshold)
        self.settings_row_2.addStretch(1)

        # Connect custom buttons and checkbox to functions
        self.zoom_to_calc_scope = False
        self.button_zoom.clicked.connect(self.__button_zoom_clicked)

        self.button_routes_line_color.clicked.connect(self.__button_routes_line_color_clicked)
        self.button_routes_marker_color.clicked.connect(self.__button_routes_marker_color_clicked)
        self.button_prediction_line_color.clicked.connect(self.__button_prediction_line_color_clicked)
        self.button_prediction_marker_color.clicked.connect(self.__button_prediction_marker_color_clicked)
        self.button_line_width.clicked.connect(self.__button_line_width_clicked)
        self.button_marker_size.clicked.connect(self.__button_marker_size_clicked)
        self.button_mark_every.clicked.connect(self.__button_mark_every_clicked)

        self.anomalies_visible = False
        self.checkbox_anomalies.stateChanged.connect(self.__checkbox_anomalies_changed)
        self.slider_anomaly_threshold.valueChanged.connect(self.__slider_anomaly_threshold_value_changed)

    def set_scope(self, scope):
        """ Sets the scope for visualization
        Parameters:
        scope: the scope object to use for visualization
        """
        self.vis.set_scope(scope)

    def draw_map(self):
        """ Draws a map in the background
        """
        self.vis.draw_map()

    def draw_routes(self):
        """ Draws the routes of the ships
        """
        self.vis.draw_routes()

    def draw_predictions(self):
        """ Draws the predictions of the ships
        """
        self.vis.draw_predictions()

    def show_dialog(self):
        """ Starts the PyQt5-Application maximized
        """
        self.showMaximized()
        sys.exit(self.app.exec_())

    def __button_zoom_clicked(self):
        """ Zooms to calculated or to user scope
        """
        self.zoom_to_calc_scope = not self.zoom_to_calc_scope
        self.vis.change_zoom(self.zoom_to_calc_scope)
        if self.zoom_to_calc_scope:
            self.button_zoom.setText("Zoom to user scope")
        else:
            self.button_zoom.setText("Zoom to calculated scope")

    def __button_routes_line_color_clicked(self):
        """ Changes the color of route lines and route patches
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.vis.change_route_line_color(color.name())

    def __button_routes_marker_color_clicked(self):
        """ Changes the color of route markers
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.vis.change_route_marker_color(color.name())

    def __button_prediction_line_color_clicked(self):
        """ Changes the color of prediction lines and prediction patches
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.vis.change_prediction_line_color(color.name())

    def __button_prediction_marker_color_clicked(self):
        """ Changes the color of prediction markers
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.vis.change_prediction_marker_color(color.name())

    def __button_line_width_clicked(self):
        """ Changes the line width and patch size
        """
        val, okPressed = QInputDialog.getDouble(self, "Set line width","Value:", self.line_thickness, 0, 100, 4)
        if okPressed:
            self.vis.change_line_width(val)
            self.line_thickness = val

    def __button_marker_size_clicked(self):
        """ Changes the marker size
        """
        val, okPressed = QInputDialog.getDouble(self, "Set marker size","Value:", self.marker_size, 0, 100, 4)
        if okPressed:
            self.vis.change_marker_size(val)
            self.marker_size = val

    def __button_mark_every_clicked(self):
        """ Shows only n-th marker
        """
        val, okPressed = QInputDialog.getInt(self, "Set every n-th marker","Value:", self.mark_every, 1, 1000)
        if okPressed:
            self.vis.change_mark_every(val)
            self.mark_every = val

    def __checkbox_anomalies_changed(self):
        """ Shows anomaly annotations
        """
        self.vis.show_anomalies(self.checkbox_anomalies.isChecked())

    def __slider_anomaly_threshold_value_changed(self):
        """ Changes the anomaly threshold and redraw the anomaly annotations, if necessary
        """
        val = self.slider_anomaly_threshold.value() / 10000
        self.label_anomaly_threshold.setText("Anomaly threshold: %.4f" % round(val, 5))
        self.vis.change_anomaly_thresh(val)
        self.vis.show_anomalies(self.checkbox_anomalies.isChecked())
        self.anomaly_threshold = val

def parse_color(color):
    """ takes the parsed color and returns it in matplotlib format
    Parameters:
    color: the color to parse
    """
    return (color[0], color[1], color[2])

def arg_parser():
    """ returns an argument parse, which parses the arugments, which are needed / optional for this component.
    """
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("visualization")
    group.add_argument("--routes_line_color", nargs=3, action="store", type=float, default=[0.6, 0.2, 0.8],
                       help="the color of the routes line", metavar=("r", "g", "b"))
    group.add_argument("--routes_marker_color", nargs=3, action="store", type=float, default=[0, 0, 1.0],
                       help="the color of the routes marker", metavar=("r", "g", "b"))
    group.add_argument("--predictions_line_color", nargs=3, action="store", type=float, default=[1.0, 0.4, 0],
                       help="the color of the predictions line", metavar=("r", "g", "b"))
    group.add_argument("--predictions_marker_color", nargs=3, action="store", type=float, default=[0, 0, 1.0],
                       help="the color of the predictions marker", metavar=("r", "g", "b"))
    group.add_argument("--marker_size", action="store", type=float, default=0.75, help="the size of the markers",
                       metavar=("size"))
    group.add_argument("--line_thickness", action="store", type=float, default=0.5, help="the thickness of the lines",
                       metavar=("width"))
    group.add_argument("--mark_every", action="store", type=int, default=5, help="mark every n-th gps-point",
                       metavar=("n"))
    group.add_argument("--anomaly_threshold", action="store", type=float, default=0.5,
                       help="the threshold probability for an anomaly", metavar="thresh")
    return parser

if __name__ == "__main__":
    from persistent.reader import Reader
    import persistent.argument

    # Parse arguments
    parser = argparse.ArgumentParser(parents=[persistent.argument.input_path_parser(), arg_parser()])
    args = parser.parse_args()

    # Create PyQt5-Application with parsed arguments
    vis = QtFrame(
       routes_line_color=parse_color(args.routes_line_color),
       routes_marker_color=parse_color(args.routes_marker_color),
       predictions_line_color=parse_color(args.predictions_line_color),
       predictions_marker_color=parse_color(args.predictions_marker_color),
       marker_size=args.marker_size,
       line_thickness=args.line_thickness,
       mark_every=args.mark_every,
       anomaly_threshold=args.anomaly_threshold)

    # Read the file information from parsed input
    reader = Reader(args.input)
    scope = reader.read()

    # Create content for plot
    vis.set_scope(scope)
    vis.draw_map()
    vis.draw_routes()
    vis.draw_predictions()
    vis.show_dialog()
