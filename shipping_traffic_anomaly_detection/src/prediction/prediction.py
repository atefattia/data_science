from numpy import array
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.merge import Concatenate
from keras.layers.merge import Multiply
from keras.models import Model
from keras.models import load_model
from src.prediction import utils
from src.aisdata.scope import Scope
from src.aisdata.ship import Ship
from src.aisdata.route import Route
from src.aisdata.state import State
import numpy as np
import argparse
from keras.utils import plot_model


class Prediction:
    n_neighbours = 2

    def __init__(self, n_features=2, input_window_size=3, training_data_size=100, forecast_size=3, n_unit=100,
                 net_type="LSTM", epochs=1000, model_save_path="", model_load_path=""):
        """Constructor of Prediction

        Parameters:
            n_features: number of features
            input_window_size: size of the input window
            training_data_size: number of training data sequences
            forecast_size: number of forecast time steps / output window
            n_unit: number of LSTM/GRU units to use per neighbour
            net_type: type of the net (LSTM or GRU)
            epochs: number of training epochs
            model_save_path: path to save model
            model_load_path: path to load model
        """
        self.nFeatures = n_features
        self.inputWindowSize = input_window_size
        self.forecastSize = forecast_size
        self.nUnit = n_unit
        self.net_type = net_type
        self.training_data_size = training_data_size
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.model_load_path = model_load_path

    def generate_model(self, X):
        """ Generate the model.

        return: model
        """
        # LSTM encoders
        inp = [Input(shape=(X[0].shape[1], X[0].shape[2])), Input(shape=(X[1].shape[1], X[1].shape[2])), Input(shape=(X[1].shape[1], X[1].shape[2]))]
        if (self.net_type == "LSTM"):
            enc = [LSTM(self.nUnit)(inp[0]), LSTM(self.nUnit)(inp[1]), LSTM(self.nUnit)(inp[2])]
        else:
            enc = [GRU(self.nUnit)(inp[0]), GRU(self.nUnit)(inp[1]), GRU(self.nUnit)(inp[2])]

        # Attention layer for observed ship
        at_probs_o = Dense(self.nUnit, activation='softmax')(enc[0])
        at_mul_o = Multiply()([enc[0], at_probs_o])

        # Attention layer for neighbours
        con_n = Concatenate()([enc[1], enc[2]])
        at_probs_n = Dense((len(X)-1)*self.nUnit, activation='softmax')(con_n)
        at_mul_n = Multiply()([con_n, at_probs_n])

        # Attention layer for all
        con_a = Concatenate()([at_mul_o, at_mul_n])
        at_probs_a = Dense(len(X)*self.nUnit, activation='softmax')(con_a)
        at_mul_a = Multiply()([con_a, at_probs_a])

        # LSTM decoder
        rep = RepeatVector(self.inputWindowSize)(at_mul_a)

        if (self.net_type == "LSTM"):
            dec = LSTM(self.nUnit, return_sequences=True)(rep)
        else:
            dec = GRU(self.nUnit, return_sequences=True)(rep)

        tdis = TimeDistributed(Dense(self.nFeatures))(dec)
        model = Model(inputs=inp, outputs=tdis)
        return model

    def train(self, X, y):
        """ Train the prediciton model

        return: trained model
        """
        #reshape from [samples, timesteps] into [samples, timesteps, features]
        for i in range(X.shape[1]):
            print("seq: " + str(i))
            print(X[0][i])
            print(X[1][i])
            print(X[2][i])
            print(y[i])

        X = list(map(lambda x: x.reshape((x.shape[0], x.shape[1], self.nFeatures)), X))
        y = y.reshape((y.shape[0], y.shape[1], self.nFeatures))

        model = self.generate_model(X)

        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=self.epochs, verbose=1)
        return model

    def predict(self, model, x_input):
        """ Predicts the next forecast_size time steps.

        return: prediction as a numpy array
        """
        # Prediction start array
        print("input:")

        x_input = x_input.reshape((3, 1, self.inputWindowSize, self.nFeatures))
        print(x_input)
        # predict
        yhat = model.predict([x_input[0], x_input[1], x_input[2]], verbose=0)
        print("output: ")
        print(yhat)

        # print result
        return yhat

    def predict_route(self, model, route, other_ships):

        X, y = utils.extract_training_examples_for_route(route, other_ships, n_ships=3, n_states=10)

        #r = route.states
        #a = []
        #for s in r:
        #    a.append([s.latitude, s.longitude])

        #a = array(a[0:self.inputWindowSize])
        print(X.shape)
        x = array([X[0][X.shape[1]-1], X[1][X.shape[1]-1], X[2][X.shape[1]-1]])
        res = self.predict(model, x)
        res = res[0]
        p_states = []
        for r in res:
            state = route.states[len(p_states) + self.inputWindowSize]
            p_states.append(State(len(p_states) + self.inputWindowSize,r[0], r[1], state.status,
                                  state.rate_of_turn, state.speed_over_ground, state.course_over_ground,
                                  state.true_heading, state.t_day, state.t_week, state.draught))

        route.predicted_states = p_states
        return route


def arg_parser():
    """ returns an argument parse, which parses the arugments, which are needed.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n_features", action='store', type=int, help="number of features", default=2)
    parser.add_argument("--input_window_size", action='store', type=int, help="size of the input window", default=3)
    parser.add_argument("--training_data_size", action='store', type=int, help="number of training sequences", default=72)
    parser.add_argument("--forecast_size", action='store', type=int, help="number of forecast time steps", default=3)
    parser.add_argument("--n_unit", action='store', type=int, help="number of LSTM / GRU units", default=100)
    parser.add_argument("--net_type", action='store', type=str, help="type of the net (LSTM | GRU)", default="LSTM")
    parser.add_argument("--epochs", action='store', type=int, help="number of training epochs", default=1000)
    parser.add_argument("--model_save_path", action='store', type=str, help="model save path", default="")
    parser.add_argument("--model_load_path", action='store', type=str, help="model load path", default="")
    return parser

if __name__ == "__main__":
    import argparse
    from src.persistent.writer import Writer
    from src.persistent.reader import Reader
    import src.persistent.argument as argument

    parser = argparse.ArgumentParser(parents=[argument.io_paths_parser(), arg_parser()])
    args = parser.parse_args()
    pr = Prediction(args.n_features, args.input_window_size, args.training_data_size, args.forecast_size, args.n_unit,
                    args.net_type, args.epochs, args.model_save_path, args.model_load_path)
    reader = Reader(args.input)
    scope = reader.read()

    ship = list(scope.ships)[0]
    other_ships = list(scope.ships)
    other_ships.remove(ship)
    route = list(ship.routes)[0]
    if(pr.model_load_path==""):
        X, y = utils.extract_training_examples_for_route(route, other_ships, n_ships=3, n_states=10)
        pr.training_data_size = X.shape[0]

        #y_values = list(map(lambda r: [r.latitude, r.longitude], route.states))
        #y = []
        #for i in range(X.shape[1]):
        #    y.append(y_values[i+pr.inputWindowSize:i+2*pr.inputWindowSize])

        #y = array(y)
        model = pr.train(X, y)
        if(pr.model_save_path != ""):
            model.save("trained_model.h5")
    else:
        model = load_model(pr.model_load_path)

    route.predicted_states = pr.predict_route(model, route, other_ships).predicted_states
    writer = Writer(args.output)
    writer.write(scope)
