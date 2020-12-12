import os
import argparse
from persistent.reader import Reader
from persistent.writer import Writer
import persistent.argument
import numpy as np
import numpy.random
import copy

import sys
assert sys.version_info >= (3, 4), "Use python3!"

parser = argparse.ArgumentParser(parents=[persistent.argument.io_paths_parser()])
parser.add_argument("--sigma", action='store', type=float, default=0.001, help="the standard derivation for the noise to apply")
args = parser.parse_args()

input_dir=os.path.abspath(args.input)
output_dir=os.path.abspath(args.output)
reader = Reader(input_dir)
scope = reader.read()

sigma = args.sigma

for ship in scope.ships:
    for route in ship.routes:
      route.predicted_states = copy.deepcopy(route.states)
      for ps in route.predicted_states:
          ps.longitude +=  np.random.normal(0, sigma, 1)[0]
          ps.latitude +=  np.random.normal(0, sigma, 1)[0]

writer = Writer(output_dir)
writer.write(scope)
