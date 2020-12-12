from aisdata.anomaly import Anomaly
import numpy as np
from numpy.linalg import inv
from scipy.spatial import distance as dist
from enum import Enum
import argparse

def clamp(value, minvalue, maxvalue):
    """ Keeps a value in the given limits.
    Parameters:
    value: the value to keep in the given limits.
    minvalue: the lower limit
    maxvalue: the upper limit
    """
    return max(minvalue, min(value, maxvalue))

def position_of(state):
    """ Returns the position (longitude, latitude) of a given ship as an numpy.array.
    Parameters:
    state: the state to extract the position from
    """
    return np.array([state.longitude, state.latitude])

def normalize_route(route):
    """ Make sure that the states and predicted_states of a route have matching number_in_route
    Parameters:
    route: the route to nomralize
    """
    route.states = [s for s in route.states if any(s.number_in_route == p.number_in_route for p in route.predicted_states)]
    route.predicted_states = [p for p in route.predicted_states if any(p.number_in_route == s.number_in_route for s in route.states)]
    return route

def normalize_scope(scope):
    """ Normalizes all routes of all ships in the given scope (see normalize_route)
    Paramters:
    scope: the scope to normalize
    """
    for ship in scope.ships:
        for route in ship.routes:
            route = normalize_route(route)
    return scope

class PredictionErrorMode(Enum):
    ABSOLUTE_PREDICTION_ERROR = 1
    RELATIVE_PREDICTION_ERROR = 2

class AnomalyDetection:
    def __init__(self, weight_matrix, min_error, max_error, error_mode = PredictionErrorMode.ABSOLUTE_PREDICTION_ERROR):
        """ Constructor of AnomalyDetection
        Parameters:
        weight_matrix: the weight matrix to use.
        min_error: the minimal error which will result in an anomaly greater or equal than 0
        max_error: the maximal error which will result in an anomaly smaller or equal than 1
        error_mode: wheather to use absolute of relative prediction error
        """
        self.weight_matrix = weight_matrix
        self.min_error = min_error
        self.max_error = max_error
        self.detection_function = self.detect_absolute_in_route if error_mode is PredictionErrorMode.ABSOLUTE_PREDICTION_ERROR\
                             else self.detect_relative_in_route

    def detect(self, scope):
        """ detects anomalies in the given scope.
        Parameters:
        scope: the scope to analysze
        """
        normalized_scope = normalize_scope(scope)
        for ship in normalized_scope.ships:
            for route in ship.routes:
                route.anomalies = self.detection_function(route)
        return normalized_scope

    def anomaly_from_error(self, error):
        """ calculates the anomaly given a specific error.
        Paramters:
        error: the error
        """
        return (clamp(error, self.min_error, self.max_error) - self.min_error)\
                / (self.max_error - self.min_error)

    def absolute_prediction_error(self, a, b):
        """ calculates the absolute prediction error of a given pair of states and the corresponding anomaly.
        Paramters:
        a: the one state
        b: the other state
        """
        error = dist.mahalanobis(position_of(a), position_of(b), self.weight_matrix)
        return Anomaly(a.number_in_route, error, self.anomaly_from_error(error))

    def detect_absolute_in_route(self, route):
        """ detects anomalies in the given route using the absolute prediction error.
        Paramters:
        route: the route.
        """
        return list(map(self.absolute_prediction_error, route.states, route.predicted_states))

    def relative_prediction_error(self, a_1, a_2, b_1, b_2):
        """ calculates the relative prediction error using the given four states and the corresponding anomaly.
        Paramters:
        a_1: a state of trajectory a
        a_2: the subsequent state of trajectory a
        b_1: a state of trajectory b
        b_2: the subsequent state of trajectory b
        """
        error = dist.mahalanobis(position_of(a_2) - position_of(a_1), position_of(b_2) - position_of(b_1), self.weight_matrix)
        return Anomaly(a_2.number_in_route, error, self.anomaly_from_error(error))

    def detect_relative_in_route(self, route):
        """ detects anomalies in the given route using the relative prediction error.
        The relative prediction error of the first state is 0 because there is no preceeding state.

        Paramters:
        route: the route.
        """
        n = len(route.states)
        if n < 2:
            return []
        return [Anomaly(0, route.states[0].number_in_route, 0)] + \
                list(map(self.relative_prediction_error, route.states[0:n-1], route.states[1:n], route.predicted_states[0:n-1], route.predicted_states[1:n]))

def transitions(states):
    """ calculates the transitions form subsequent states.
    The following condition holds: state[i] = transition[i] + state[i - 1] for i > 0.
    Paramters:
    states: the states to calculate the transitions from.
    """
    n = len(states)
    if n < 2:
        return []
    return [position_of(a) - position_of(b) for (a, b) in zip(states[0:n-1], states[1:n])]

def absolute_differences(routes):
    """ calculates the difference between the states and the corresponding predictions.
    'absolute' means in this context, that the absolute positions are used (in opposition to the relative position).
    The difference is stell a vector and is signed.
    """
    return [position_of(state) - position_of(predicted_state) for route in routes\
                                                              for (state, predicted_state) in zip(route.states, route.predicted_states)]

def relative_differences(routes):
    """ calculates the relative differences between subsequent states and their curresponding predictions.
    Paramters:
    route: the route whose state and predicted_states should be subtracted. 
    """
    return [state_transition - predicted_state_transition for route in routes\
                                                          for (state_transition, predicted_state_transition) in zip(transitions(route.states), transitions(route.predicted_states))]

def estimate_covariance_from_scope(scope, error_mode = PredictionErrorMode.ABSOLUTE_PREDICTION_ERROR):
   """ estimates the covariance of the prediction error in a given scope.
   Paramters:
   scope: the scope to get the covariance of.
   error_mode: wheather to use absolute of relative prediction error
   """
   normalized_scope = normalize_scope(scope)
   normalized_routes = [route for ship in normalized_scope.ships for route in ship.routes]
   diffs = absolute_differences(normalized_routes) if error_mode is PredictionErrorMode.ABSOLUTE_PREDICTION_ERROR else\
           relative_differences(normalized_routes)
   assert len(diffs) >= 2, "we at least 2 elements for covariance calculation"
   return np.cov(np.transpose(diffs))

def as_error_mode(s):
  if s == 'relative':
     return PredictionErrorMode.RELATIVE_PREDICTION_ERROR
  if s == 'absolute':
     return PredictionErrorMode.ABSOLUTE_PREDICTION_ERROR
  return None

def arg_parser_error_mode():
    """ returns an argument parser for the error mode.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--error_mode", action='store', type=as_error_mode, default=PredictionErrorMode.ABSOLUTE_PREDICTION_ERROR , help="the error mode to use. 'absolute' or 'relative' are possible. 'absolute' is the default.")
    return parser

def arg_parser():
    """ returns an argument parser, which parses the arugments, which are needed for this component.
    """
    parser = argparse.ArgumentParser(add_help=False, parents=[arg_parser_error_mode()])
    group = parser.add_argument_group('anomaly detection')
    group.add_argument("--min_error", action='store', type=float, default=1, help="the minimal error which will result in an anomaly greater or equal than 0")
    group.add_argument("--max_error", action='store', type=float, default=2, help="the maximal error which will result in an anomaly smaller or equal than 1")
    w_group =  group.add_mutually_exclusive_group(required=True)
    w_group.add_argument("--weights", action='store', nargs=2, default=(1, 1),  metavar=('weight_longitude', 'weight_latitude'), help="the weights, which should be used in mahalanobis-distance")
    w_group.add_argument("--covariance", action='store', nargs=4, default=(1, 0, 0, 1),  metavar=('c_xx', 'cxy', 'c_yx', 'c_yy'), help="the covariance, which should be used in mahalanobis-distance")
    return parser

if __name__ == "__main__":
   from persistent.reader import Reader
   from persistent.writer import Writer
   from argparse import RawTextHelpFormatter
   import persistent.argument

   description ="\
Anomaly detections purpose is to decet anomalies. An anomaly is a point in time \n\
where the predicted and the measured state differe significantly. For measuring \n\
the difference, there are two options possible here: relative and absolute error.\n\
The absolute error compares the predicted and the measured states as they are.\n\
The relative error does not compare the states but the transitions between them \n\
and their direct predecessor. \n\n\
Because the prediction well usually not be perfect, it seems reasonable to have \n\
a measurement of the error, which takes into account the uncertainties of the \n\
prediction. In anomaly detection this is archieved by using the mahalanobis-\n\
distance. This requires a covariance-matrix or a weight-matrix. The covariance-\n\
matrix can be obtained by running estimate_covariance.py on a data set.\n\n\
By appling the mahalanobis-distance to the absolute/relative error we get floating\n\
point value greater zero, which indicates how unusual the error is. But we want to\n\
get a value between zero and one to indicate, how anormal it is. Therefore we\n\
introduce min_error and max_error. For an error smaller than min_error, the \n\
anomaly score will be 0, for an error greeater than max_error, the anomaly  score\n\
will be 1. Inbetween, the values will be interpolated linearly. \n\n\
The error and the anomaly score for every predicted state will be stored in the\n\
output files.\
"

   parser = argparse.ArgumentParser(parents=[persistent.argument.io_paths_parser(), arg_parser()], description=description, formatter_class=RawTextHelpFormatter)
   args = parser.parse_args()
   weight_matrix = np.diag(args.weights) if args.weights else \
                   inv(np.array([args.covariance[0:1], args.covariance[2:3]]))
   ad = AnomalyDetection(weight_matrix, args.min_error, args.max_error, args.error_mode)
   reader = Reader(args.input)
   scope = reader.read()
   scope_with_anomalies = ad.detect(scope)
   writer = Writer(args.output)
   writer.write(scope_with_anomalies)
