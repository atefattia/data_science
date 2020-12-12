import os
import unittest
import numpy as np
import anomaly_detection.anomaly_detection as anomaly_detection
from anomaly_detection.anomaly_detection import AnomalyDetection
from anomaly_detection.anomaly_detection import PredictionErrorMode 

class ScopeStub:
    def __init__(self, ships):
        self.ships = ships

class ShipStub:
    def __init__(self, routes):
        self.routes = routes

class RouteStub:
    def __init__(self, states, predicted_states=None):
        self.states = states
        self.predicted_states = predicted_states if predicted_states is not None else states

class StateStub:
    def __init__(self, number_in_route, longitude, latitude):
        self.number_in_route = number_in_route
        self.longitude = longitude
        self.latitude = latitude

class TestAnomalyDetection(unittest.TestCase):
   def test_no_ships_result_in_empty_anomalies(self):
       ad = AnomalyDetection(np.identity(2), 0, 1)
       scope = ScopeStub([])
       scope_with_anomalies = ad.detect(scope)
       self.assertEqual(0, len(scope_with_anomalies.ships))

   def test_no_routes_results_in_empty_anomalies(self):
       ad = AnomalyDetection(np.identity(2), 0, 1)
       scope = ScopeStub([ShipStub([])])
       scope_with_anomalies = ad.detect(scope)
       self.assertEqual(1, len(scope_with_anomalies.ships))
       self.assertEqual(0, len(scope_with_anomalies.ships[0].routes))

   def test_no_states_results_in_empty_anomalies(self):
       ad = AnomalyDetection(np.identity(2), 0, 1)
       scope = ScopeStub([ShipStub([RouteStub([])])])
       scope_with_anomalies = ad.detect(scope)
       self.assertEqual(1, len(scope_with_anomalies.ships))
       self.assertEqual(1, len(scope_with_anomalies.ships[0].routes))
       self.assertEqual(0, len(scope_with_anomalies.ships[0].routes[0].states))

   def test_perfect_predictions_are_normal_absolute(self):
       ad = AnomalyDetection(np.identity(2), 0, 1)
       scope = ScopeStub([ShipStub([RouteStub([StateStub(0,0,0), StateStub(1,1,4)])])])
       scope_with_anomalies = ad.detect(scope)
       self.assertEqual(1, len(scope_with_anomalies.ships))
       self.assertEqual(1, len(scope_with_anomalies.ships[0].routes))
       self.assertEqual(2, len(scope_with_anomalies.ships[0].routes[0].anomalies))
       for a in scope_with_anomalies.ships[0].routes[0].anomalies:
           self.assertEqual(0, a.error)

   def test_perfect_predictions_are_normal_relative(self):
       ad = AnomalyDetection(np.identity(2), 0, 1, error_mode=PredictionErrorMode.RELATIVE_PREDICTION_ERROR)
       scope = ScopeStub([ShipStub([RouteStub([StateStub(0,0,0), StateStub(1,1,4)])])])
       scope_with_anomalies = ad.detect(scope)
       self.assertEqual(1, len(scope_with_anomalies.ships))
       self.assertEqual(1, len(scope_with_anomalies.ships[0].routes))
       self.assertEqual(2, len(scope_with_anomalies.ships[0].routes[0].states))
       for a in scope_with_anomalies.ships[0].routes[0].anomalies:
           self.assertEqual(0, a.error)

#   def test_covariance_estimation_absolute(self):
#       scope = ScopeStub([ShipStub([RouteStub([StateStub(0,0,0), StateStub(1,1,4), StateStub(2,3,4)])])])
#       cov = anomaly_detection.estimate_covariance_from_scope(scope)
#       self.assertEqual(2, cov.shape[0])
#       self.assertEqual(2, cov.shape[1])

#   def test_covariance_estimation_absolute(self):
#       scope = ScopeStub([ShipStub([RouteStub([StateStub(0,0,0), StateStub(1,1,4), StateStub(2,3,4), StateStub(3,9,0)])])])
#       cov = anomaly_detection.estimate_covariance_from_scope(scope, error_mode=PredictionErrorMode.RELATIVE_PREDICTION_ERROR)
#       self.assertEqual(2, cov.shape[0])
#       self.assertEqual(2, cov.shape[1])


if __name__ == '__main__':
    unittest.main()
