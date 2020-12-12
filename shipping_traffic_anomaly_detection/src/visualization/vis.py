
""" visualization for the RNN """

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature, COLORS
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import math

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from visualization.ship_type_dict import ship_type_dict

class Visualization(FigureCanvas):
    def __init__(
            self,
            routes_line_color=(0.6, 0.2, 0.8),
            routes_marker_color=(0, 0, 1.0),
            predictions_line_color=(1.0, 0.4, 0.0),
            predictions_marker_color=(0, 0, 1.0),
            marker_size=0.75,
            line_thickness=0.5,
            mark_every=5,
            anomaly_threshold=0.5):
        """ Constructor of Visualization
        Parameters:
        routes_line_color: the color for route lines
        routes_marker_color: the color for route markers
        predictions_line_color: the color for prediction lines
        predictions_marker_color: the color for prediction markers
        marker_size: the size of markers
        line_thickness: the width of lines
        mark_every: the number to show only every n-th marker
        """
        self.routes_line_color = routes_line_color
        self.routes_marker_color = routes_marker_color
        self.predictions_line_color = predictions_line_color
        self.predictions_marker_color = predictions_marker_color
        self.marker_size = marker_size
        self.line_thickness = line_thickness
        self.mark_every = mark_every
        self.anomaly_threshold = anomaly_threshold

        # Create subplot with PlateCarree projection.
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection=ccrs.PlateCarree())

        FigureCanvas.__init__(self, self.fig)

        # Label axes
        plt.xlabel('degree of longitude')
        plt.ylabel('degree of latitude')

        # Create annotation for line annotations
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"), zorder=100)
        self.annot.set_visible(False)

        self.anomalies_annotations = []

        self.arrows = []
        self.arrow_routes_positions = []
        self.arrow_routes_directions = []
        self.arrow_predictions_positions = []
        self.arrow_predictions_directions = []

        # Adjust margin of plot
        plt.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.97)

        self.set_to_calc_scope = False

        # Connect events with functions
        self.fig.canvas.mpl_connect('motion_notify_event', self.__on_hover)
        self.ax.callbacks.connect('xlim_changed', self.__on_xlims_change)
        self.ax.callbacks.connect('ylim_changed', self.__on_ylims_change)
        self.__change_sizes()

    def change_route_line_color(self, color):
        """ Changes the color of route lines and route patches
        Parameters:
        color: the new color
        """
        self.routes_line_color = color
        for curve in self.ax.get_lines():
            if curve.get_gid() == "r":
                plt.setp(curve, color=color)
        for arrow in self.arrows:
            if arrow.get_gid() == "r":
                arrow.set_color(color)
        self.fig.canvas.draw()

    def change_route_marker_color(self, color):
        """ Changes the color of route markers
        Parameters:
        color: the new color
        """
        self.routes_marker_color = color
        for curve in self.ax.get_lines():
            if curve.get_gid() == "r":
                plt.setp(curve, markerfacecolor=color, markeredgecolor=color)
        self.fig.canvas.draw()

    def change_prediction_line_color(self, color):
        """ Changes the color of prediction lines and prediction patches
        Parameters:
        color: the new color
        """
        self.predictions_line_color = color
        for curve in self.ax.get_lines():
            if curve.get_gid() == "p":
                plt.setp(curve, color=color)
        for arrow in self.arrows:
            if arrow.get_gid() == "p":
                arrow.set_color(color)
        self.fig.canvas.draw()

    def change_prediction_marker_color(self, color):
        """ Changes the color of prediction markers
        Parameters:
        color: the new color
        """
        self.predictions_marker_color = color
        for curve in self.ax.get_lines():
            if curve.get_gid() == "p":
                plt.setp(curve, markerfacecolor=color, markeredgecolor=color)
        self.fig.canvas.draw()

    def change_line_width(self, val):
        """ Changes the line width and patch size
        Parameters:
        val: the new value
        """
        self.line_thickness = val
        for curve in self.ax.get_lines():
            plt.setp(curve, linewidth=val)
        self.__clear_directions()
        for i in range(0, len(self.arrow_routes_positions)):
            pos = self.arrow_routes_positions[i]
            dir = self.arrow_routes_directions[i]
            self.__draw_direction(pos, dir, self.routes_line_color, "r")
        for i in range(0, len(self.arrow_predictions_positions)):
            pos = self.arrow_predictions_positions[i]
            dir = self.arrow_predictions_directions[i]
            self.__draw_direction(pos, dir, self.predictions_line_color, "p")
        self.__change_sizes()
        self.fig.canvas.draw()

    def change_marker_size(self, val):
        """ Changes the marker size
        Parameters:
        val: the new value
        """
        self.marker_size = val
        for curve in self.ax.get_lines():
            plt.setp(curve, ms=val)
        self.__change_sizes()
        self.fig.canvas.draw()

    def change_mark_every(self, val):
        """ Shows only n-th marker
        Parameters:
        val: the new value
        """
        self.mark_every = val
        for curve in self.ax.get_lines():
            plt.setp(curve, markevery=val)
        self.fig.canvas.draw()

    def __on_hover(self, event):
        """ Shows ship label, when the cursor is over line or patch
        Parameters:
        event: the given event object for the motion notify event
        """
        vis = False
        text = ""
        # Check if cursor is over line or patch
        for curve in self.ax.get_lines():
            if curve.contains(event)[0]:
                vis = True
                text = curve.get_label()
        for arrow in self.arrows:
            if arrow.contains(event)[0]:
                vis = True
                text = arrow.get_label()
        # If cursor is over line or patch, show annotation
        if vis:
            self.annot.set_text(text)
            self.annot.xy = (event.xdata, event.ydata)
            self.annot.set_visible(True)
        else:
            self.annot.set_visible(False)
        self.fig.canvas.draw_idle()

    def __on_xlims_change(self, axes):
        """ Changes axes ticker and line and patch sizes when xlim changes
        Parameters:
        axes: the given axes object for the given change xlim event
        """
        self.__change_ticker()
        self.__change_sizes()

    def __on_ylims_change(self, axes):
        """ Changes axes ticker and line and patch sizes when ylim changes
        Parameters:
        axes: the given axes object for the given change ylim event
        """
        self.__change_ticker()
        self.__change_sizes()

    def __change_ticker(self):
        """ Changes axes ticker
        """
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        xlim_low = round(xlim[0], 2)
        xlim_high = round(xlim[1], 2)
        ylim_low = round(ylim[0], 2)
        ylim_high = round(ylim[1], 2)

        """ Set high ratio, if ylim_high - ylim_low == 0
        Otherwise calculate ratio of xlim to ylim
        """
        if ylim_high - ylim_low == 0:
            ratio = 500000
        else:
            ratio = (xlim_high - xlim_low) / (ylim_high - ylim_low)

        # Calculate number of tickers regarding calculated ratio
        if ratio > 1:
            num_x_ticks = 10
            num_y_ticks = max(10 / ratio, 1)
        else:
            num_x_ticks = max(10 * ratio, 1)
            num_y_ticks = 10

        # Calculate steps for ticker
        longitude_step = round((xlim_high - xlim_low) / num_x_ticks, 2)
        latitude_step = round((ylim_high - ylim_low) / num_y_ticks, 2)

        # Set the ticker regarding steps and ratio
        if longitude_step == 0 or ratio < 0.2:
            self.ax.set_xticks([round((xlim_high + xlim_low) / 2, 2)])
        else:
            self.ax.set_xticks(
                np.arange(xlim_low,
                      xlim_high + longitude_step,
                      longitude_step),
                crs=ccrs.PlateCarree())

        if latitude_step == 0 or ratio > 10:
            self.ax.set_yticks([round((ylim_high + ylim_low) / 2, 2)])
        else:
            self.ax.set_yticks(
                np.arange(ylim_low,
                      ylim_high + latitude_step,
                      latitude_step),
                crs=ccrs.PlateCarree())

    def __change_sizes(self):
        """ Changes sizes of lines and markers regarding zoom level
        """
        # Calculate average of xlim and ylim different
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        xlim_low = xlim[0]
        xlim_high = xlim[1]
        ylim_low = ylim[0]
        ylim_high = ylim[1]

        average = (xlim_high - xlim_low + ylim_high - ylim_low) / 2

        # Calculate sizes and set them
        ms = self.marker_size / average
        linewidth = self.line_thickness / average

        for curve in self.ax.get_lines():
            plt.setp(curve, linewidth=linewidth, ms= ms, markevery=self.mark_every)

    def change_zoom(self, zoom_to_calc_scope):
        """ Zooms to calculated or to user scope
        Parameters:
        zoom_to_calc_scope: bool, if zoom to calculated scope
        """
        if zoom_to_calc_scope:
            s = [self.scope.longitude_min, self.scope.longitude_max,
                 self.scope.latitude_min, self.scope.latitude_max]
        else:
            s = [self.scope.longitude_min_user, self.scope.longitude_max_user,
                 self.scope.latitude_min_user, self.scope.latitude_max_user]
        self.ax.axis(s)
        self.fig.canvas.draw()

    def __get_ship_label(self, ship):
        """ Returns label of ships
        Parameters:
        ship: the ship for creating label
        """
        label = 'ship name: ' + ship.shipname + '\n\n'
        label = label + ship_type_dict[0] + "\n"
        label = label + ship_type_dict[ship.shiptype]
        return label

    def __create_anomaly_annotation(self, pos, label):
        """ Creates the annotations for anomalies
        """
        annot = self.ax.annotate(label, xy=(pos[0], pos[1]), xytext=(5,5), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="r"), transform=ccrs.PlateCarree(), zorder=99)
        annot.set_visible(False)
        self.anomalies_annotations.append(annot)

    def set_scope(self, scope):
        """ Sets the scope for visualization, zooms to user scope and changes ticker
        Parameters:
        scope: the scope object to use for visualization
        """
        self.scope = scope

        # Save gap between min and max for longitude and latitude to calculate state positions of ships
        self.longitude_diff = scope.longitude_max - scope.longitude_min
        self.latitude_diff = scope.latitude_max - scope.latitude_min

        s = [scope.longitude_min_user, scope.longitude_max_user,
             scope.latitude_min_user, scope.latitude_max_user]

        # Set format of x- and y-axes
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        self.ax.xaxis.set_major_formatter(lon_formatter)
        self.ax.yaxis.set_major_formatter(lat_formatter)

        # Zoom to user scope
        self.ax.axis(s)

        # Change ticker
        self.__change_ticker()

    def draw_map(self):
        """ Draws a map in the background
        """
        land = NaturalEarthFeature(name='land', category='physical', scale='50m', facecolor=COLORS['land'])
        ocean = NaturalEarthFeature(name='ocean', category='physical', scale='50m', facecolor=COLORS['water'])

        self.ax.add_feature(land)
        self.ax.add_feature(ocean)

    def __draw_direction(self, pos, dir, color, label, gid):
        """ Draws patches to show directions
        Parameters:
        pos: the position of the patch
        dir: the direction of the patch
        color: the color of the patch
        label: the label of the patch, is used for annotations
        gid: the gid of the patch, is used to decide if patch is route or prediction patch
        """
        # Calculate width and length of patch
        width = self.line_thickness / 100
        length = width * 6

        # normalize direction and calculate normal vector
        s = math.sqrt(dir[0] * dir[0] + dir[1] * dir[1])
        dir[0] = dir[0] / s
        dir[1] = dir[1] / s
        n = [-dir[1], dir[0]]

        # if patch is prediction patch, use higher zorder
        zorder = 10
        if gid == "p":
            zorder = 22

        # Create patches
        arrow = mpatches.Polygon([[pos[0] + length * dir[0], pos[1] + length * dir[1]],
                                  [pos[0] + width * n[0], pos[1] + width * n[1]],
                                  [pos[0] - width * n[0], pos[1] - width * n[1]]],
                                 facecolor=color,
                                 transform=ccrs.PlateCarree(),
                                 label=label,
                                 gid=gid,
                                 zorder=zorder)
        self.arrows.append(arrow)
        self.ax.add_patch(arrow)

    def __clear_directions(self):
        """ Deletes all patches to recreate patches with other size
        """
        for arrow in self.arrows:
            arrow.remove()
        self.arrows = []

    def draw_routes(self):
        """ Draws lines, markers and patches for routes for ships
        """
        for ship in self.scope.ships:
            ship_label = self.__get_ship_label(ship)
            num_routes = len(ship.routes)
            for i, route in enumerate(ship.routes):
                longitudes = []
                latitudes = []

                # Calculate longitudes and latitudes from normalized values
                for state in route.states:
                    longitudes.append(self.scope.longitude_min + state.longitude * self.longitude_diff)
                    latitudes.append(self.scope.latitude_min + state.latitude * self.latitude_diff)

                l = len(longitudes)

                if l < 2:
                    continue

                # Plot lines and markers
                self.ax.plot(
                    longitudes,
                    latitudes,
                    '-o',
                    color=self.routes_line_color,
                    markerfacecolor=self.routes_marker_color,
                    markeredgecolor=self.routes_marker_color,
                    ms=self.marker_size,
                    linewidth=self.line_thickness,
                    label=ship_label,
                    gid="r")

                # Draw patches for directions, if route is last route of ship
                if i == num_routes - 1:
                    self.__draw_direction([longitudes[l - 1], latitudes[l - 1]],
                                          [longitudes[l - 1] - longitudes[l - 2], latitudes[l - 1] - latitudes[l - 2]],
                                          self.routes_line_color,
                                          ship_label,
                                          "r")
                    self.arrow_routes_positions.append([longitudes[l - 1],
                                                         latitudes[l - 1]])
                    self.arrow_routes_directions.append([longitudes[l - 1] - longitudes[l - 2],
                                                         latitudes[l - 1] - latitudes[l - 2]])
        self.__change_sizes()

    def draw_predictions(self):
        """ Draws lines, markers and patches for predictions for ships
        """
        for ship in self.scope.ships:
            ship_label = self.__get_ship_label(ship)
            for route in ship.routes:
                longitudes = []
                latitudes = []

                # Calculate longitudes and latitudes from normalized values and create anomaly annotations
                for i, state in enumerate(route.predicted_states):
                    longitude = self.scope.longitude_min + state.longitude * self.longitude_diff
                    latitude = self.scope.latitude_min + state.latitude * self.latitude_diff
                    longitudes.append(longitude)
                    latitudes.append(latitude)
                    if i < len(route.anomalies):
                        self.__create_anomaly_annotation([longitude, latitude],
                                                      "Anomaly probability: " + str(round(route.anomalies[i], 4)))

                l = len(longitudes)

                if l < 2:
                    continue

                # Plot lines and markers
                self.ax.plot(
                    longitudes,
                    latitudes,
                    '-o',
                    color=self.predictions_line_color,
                    markerfacecolor=self.predictions_marker_color,
                    markeredgecolor=self.predictions_marker_color,
                    ms=self.marker_size,
                    linewidth=self.line_thickness,
                    label=ship_label,
                    gid="p",
                    zorder=11)

                # Draw patches for directions
                self.__draw_direction([longitudes[l - 1], latitudes[l - 1]],
                                      [longitudes[l - 1] - longitudes[l - 2], latitudes[l - 1] - latitudes[l - 2]],
                                      self.predictions_line_color,
                                      ship_label,
                                      "p")
                self.arrow_predictions_positions.append([longitudes[l - 1],
                                                          latitudes[l - 1]])
                self.arrow_predictions_directions.append([longitudes[l - 1] - longitudes[l - 2],
                                                          latitudes[l - 1] - latitudes[l - 2]])
        self.__change_sizes()

    def change_anomaly_thresh(self, thresh):
        # Changes the anomaly threshold
        self.anomaly_threshold = thresh

    def show_anomalies(self, draw):
        """ Shows or hides anomaly annotations
        Parameters:
        draw: bool, if anomalies should be shown
        """
        for annot in self.anomalies_annotations:
            val = annot.get_text()
            val = float(val[21:])
            if val > self.anomaly_threshold:
                annot.set_visible(draw)
            else:
                annot.set_visible(False)

        self.fig.canvas.draw_idle()
