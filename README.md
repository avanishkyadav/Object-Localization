# Object-Localization

This project uses a 56 layers residual network to localize an object in an image. 

## Dependencies
* Numpy
* Pandas
* OpenCV
* Keras
* Matplotlib


## Dataset
https://www.kaggle.com/asquare14/imagedata

Dataset contains 640 x 480 sized images, each of them containing a single object which we need to localize.

## Preprocessing
Firstly,to account for the computational constraint we resized the images with the help of OpenCV to a size of 120 x 90 and
stored it as numpy array in .npy format.

Whole dataset is then shuffled and splited into training and testing set in ratio of 9:1.

## Training
During training with default hyperparameters we saw a large fluctutaions in accuracy. So after hypertuning we reduced
the learning_rate to **0.00001** with a decay of **0.00002** .

## Accuracy
 After model was trained for 150 epochs, we got an IOU (Intersection Over Union) accuracy of 96 and 91 percent on training set and
 testing set respectively.

## Improvement
To reduce overfitting
* Dropout
* L1 and L2 Regularisations


