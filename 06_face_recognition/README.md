# FACE RECOGNITION TASK

## DESCRIPTION
In  this  task  you  will  implement a  fun  application.
You  will  build  a  model  to learn from a bunch of face images from celebrities.
Then your model must tell whom a given image containing a person face is most similar to.
Please note that each celebrity has only several images to train (most of them have only one image).

## DATASET
http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz

You might use other information of the dataset/task at:
http://vis-www.cs.umass.edu/lfw/#deepfunnel-anchor

## REQUIREMENTS
- Build the neural face recognition model.
- Test your model on some face images to say the name of the celebrity most similar to the image.

## EVALUATION
There is a testset in the dataset, and you can evaluate it using precision/recall or accuracy.


## REFERENCES
Gary B. Huang, Marwan Mattar, Honglak Lee, and Erik Learned-Miller.
Learning to Align from Scratch.
Advances in Neural Information Processing Systems (NIPS), 2012.
http://vis-www.cs.umass.edu/papers/nips2012_deep_congealing.pdf


# SETUP

- clone the repo to your machine (paperspace)
- run the `data_expolation.ipynb` script, this will download the dataset and
  generate some auxiliary csv tables
- use either `classifier_dataset.ipynb` or `eval_representations.ipynb`
  to train a model and export it
- use the trained model in `prediction.ipynb` to visualize some predictions
