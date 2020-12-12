from persistent.reader import Reader
from persistent.writer import Writer
import persistent.argument
from anomaly_detection import estimate_covariance_from_scope
from anomaly_detection import arg_parser_error_mode 

import argparse
import math

description ="estimate_covariance.py estimates the covariance of the specified error based on the given data set."
parser = argparse.ArgumentParser(parents=[persistent.argument.input_path_parser(), arg_parser_error_mode()], description=description)
args = parser.parse_args()
reader = Reader(args.input)
scope = reader.read()

cov = estimate_covariance_from_scope(scope)

print("covariance:")
print(str(cov[0, 0]) + " " + str(cov[0, 1]) + " " + str(cov[1, 0]) + " " + str(cov[1, 1]))
print("stddevs: " + str(math.sqrt(cov[0,0])) + ", " + str(math.sqrt(cov[1,1])))
