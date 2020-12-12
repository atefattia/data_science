# BAFPracticalCourse-Code
BAF (Bildauswertung und -fusion) Practical Course Code

## Python Version

Python 3.5.6

## Install requirements

    sudo apt-get install proj-bin libproj-dev

    pip3 install -r requirements.txt

    cd setup/cartopy-0.17.0

    python3 setup.py install

## Use Docker for running code

Creating docker image:

    sudo docker build . -t pbaf

Run script inside docker:

    sudo docker run -it --rm -v "$PWD":/home pbaf python3 test/test_ais_import.py

## Usage

### Prerequisites

Execute

    source setup.bash

to setup environment.

### Usage

To use this program you have to perform these commands in the shown order.
Some of the arguments are optional, so you don't have to use every argument.

#### Data Import

Use

```bash
python3.5 src/data_import/aisimport.py --nari-dir "<path_to_nari_dircetory>" \
"<path_to_save_preprocessed_data>"
```

for importing csv files and

```bash
python3.5 src/data_import/aisimport.py --longitude_min -7 \
--longitude_max -2 --latitude_min 40 --latitude_max 50 \
--t_min 0 --t_max 1600000000 --url <url_to_database> --db ais \
--user <database_username> --password <database_password> \
"<path_to_save_preprocessed_data>"
```
to use the ais database at IOSB.

##### Arguments

See [Preprocessing](#Preprocessing)

###### url
The url to connect to database.

Type: string

###### db
The name of the database.

Type: string

Note: should be "ais"

###### user
The user name to use for access.

Type: string

###### password
The user password to use for access.

Type: string


#### Preprocessing

Use

```bash
python3.5 src/data_preprocessing/preprocessing.py --longitude_min -7 \
--longitude_max -2 --latitude_min 40 --latitude_max 50 \
--t_min 0 --t_max 1600000000 --nari-dir "<path_to_nari_dircetory>" \
"<path_to_save_preprocessed_data>"
```

to preprocess the data for prediction step.

##### Arguments

###### longitude_min
The most western longitude to use.

Type: float

Requirements: >= -180.0

Note:

###### longitude_max
The most eastern longitude to use.

Type: float

Requirements: > longitude_min && <= 180.0

Note:

###### latitude_min
The most southern latitude to use.

Type: float

Requirements: >= -90.0

Note:

###### latitude_max
The most northern latitude to use.

Type: float

Requirements: > latitude_min && <= 90.0

Note:

###### t_min (optional)
The start of time horizont (Unix timestamp).

Type: int

Requirements: >= 0

Note:

###### t_max (optional)
The end of time horizont (Unix timestamp).

Type: int

Requirements: > t_min

Note:

###### nari-dir
The directory, which contains the NARI-dataset.

Type: str

Requirements:

Note:

#### Prediction

Use

```bash
python3.5 src/prediciton/prediction.py  --n_features 2 \
--input_window_size 10 --training_data_size 100 \
--forecast_size 5 --n_unit 100 --net_type "LSTM"\
--epochs=1000 "<path_to_load_data>" "<path_to_save_data>"
```

to calculate the predicted states.

Predictions purpose is to predict the routes. Therefore at first the neural net is
trained with the given data. Then the next time steps of every route in the given
scope are predicted by the neural network.

##### Arguments

###### n_features (optional)
The number of features.

Type: int

Requirements: has to be not less than two.

###### input_window_size (optional)
The size of the input window is the number of time steps given in one training sequence.
For prediction the same number of time steps is required

Type: int

Requirements: has to be not less than two.

###### training_data_size (optional)
The number of training sequences.

Type: int

Requirements: has to be not less than one.

###### forecast_size (optional)
The number of forecast time steps. For training they must be given for every
sequence.

Type: int

Requirements: has to be not less than one.

###### n_unit (optional)
The number of LSTM or GRU units.

Type: int

Requirements: has to be not less than one.

###### net_type (optional)
Type of the used neural network

Type: str

Requirements: has to be either "LSTM" or "GRU".

###### epochs (optional)
Number of training epochs

Type: int


#### Anomaly Detection

```bash
python3.5 src/anomaly_detection/anomaly_detection.py  --weights 1 1 \
--error_mode absolute \
"<path_to_load_data>" "<path_to_save_data>"
```

Anomaly detections purpose is to decet anomalies. An anomaly is a point in time
where the predicted and the measured state differ significantly. For measuring
the difference, there are two options possible here: relative and absolute error.
The absolute error compares the predicted and the measured states as they are.
The relative error does not compare the states but the transitions between them
and their direct predecessor.

Because the prediction will usually not be perfect, it seems reasonable to have
a measurement of the error, which takes into account the uncertainties of the
prediction. In anomaly detection this is achieved by using the mahalanobis-
distance. This requires a covariance-matrix or a weight-matrix. The covariance-
matrix can be obtained by running estimate_covariance.py on a data set.
By applying the mahalanobis-distance to the absolute/relative error we get floating
point value greater zero, which indicates how unusual the error is. But we want to
get a value between zero and one to indicate, how abnormal it is. Therefore we
introduce min_error and max_error. For an error smaller than min_error, the
anomaly score will be 0, for an error greater than max_error, the anomaly score
will be 1. Inbetween, the values will be interpolated linearly.
The error and the anomaly score for every predicted state will be stored in the
output files.

##### Arguments

###### min_error (optional)
The minimal error, which will result in an anomaly greater or equal than 0.

Type: float

Requirements: has to be not less than zero and smaller than max_error.

###### max_error (optional)
The maximal error, which will result in an anomaly smaller or equal than 1.

Type: float

Requirements: has to be greater than min_error.

###### weights
The weights in longitude and latitude directions which will be used for mahalanobis-distance.

Type: (float, float)

Requirements: both values have to be not negative.

###### covariance
The covariance matrix which will be used for mahalanobis-distance.

Type: (float, float, float, float)

Requirements: the matrix needs to be symmetrical and positive definite.

Note: The matrix must be given in the form: c_xx, cxy, c_yx, cyy.

###### error_mode (optional)
The type to calculate errors.

Type: "relative" or "absolute"

#### Visualization

Use

```bash
python3.5 src/visualization/qt_frame.py --routes_line_color 0.6 0.2 0.8 \
--routes_marker_color 0.0 0.0 1.0 --predictions_line_color 1.0 0.4 0.0 \
--predictions_marker_color 0.0 0.0 1.0 --marker_size 0.75 \
--line_thickness 0.5 --mark_every 5 --anomaly_threshold 0.5 \
"<path_to_saved_data>"
```

to start the visualization application. It is reading the saved data and showing
it inside a PyQt5-application. Customize the representation using the arguments
or directly inside the application.

##### Arguments

###### routes_line_color (optional)
The color of the routes line.

Type: float float float

Requirements: 0.0 <= _val_ <= 1.0

Note: Order of values are in rgb.

###### routes_marker_color (optional)
The color of the routes marker.

Type: float float float

Requirements: 0.0 <= _val_ <= 1.0

Note: Order of values are in rgb.

###### predictions_line_color (optional)
The color of the predictions line.

Type: float float float

Requirements: 0.0 <= _val_ <= 1.0

Note: Order of values are in rgb.

###### predictions_marker_color (optional)
The color of the predictions marker.

Type: float float float

Requirements: 0.0 <= _val_ <= 1.0

Note: Order of values are in rgb.

###### marker_size (optional)
The size of the markers.

Type: float

Requirements: >= 0.0

##### line_thickness (optional)
The thickness of the lines.

Type: float

Requirements: >= 0.0

##### mark_every (optional)
Mark every n-th gps-point.

Type: int

Requirements: > 0

###### anomaly_threshold (optional)
The threshold probability for an anomaly.

Type: float

Requirements: 0.0 <= _val_ <= 1.0
