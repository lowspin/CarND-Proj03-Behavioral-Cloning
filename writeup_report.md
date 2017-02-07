#**Behavioral Cloning Project** 

##Background

This repository contains my project report for the [Udacity Self-Driving Car nano-degree](https://www.udacity.com/drive) program's project 3 - Behavioral Cloning. The original starting files and instructions can be found [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3.git). Sample training [data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by udacity were used, along with udacity's [driving simulator](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip). The project was done in Ubuntu 16.04 LTS using local Nvidia GTX960M GPU.

---

## Goals

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./res/keras_model_summary.png "Model Summary"
[image2]: ./res/driving-data-stats.png "Visualize udacity's drivingn data"
[image3]: ./res/left_center_right_camera_images.png "Sample left, center and right camera images"
[image4]: ./res/example_pre-processing.png "Sample pre-processing for each frame"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py - main script to create and train the model
* mytools.py - containing miscellaneous visualization functions
* drive.py - for driving the car in autonomous mode
* model.h5 - containing a trained convolution neural network 
* writeup_report.md (this file) - summarizing the results
* /res/run1.mp4 - video generated from autonomous driving around track 1 (in case evaluation fails).

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. Keras's ImageDataGenerator was used with fit_generator to batch-generate training data rather than storing the training data in memory. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model is based on the original Nvidia paper ["End to End Learning for Self-Driving Cars"](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 
The first part consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 175-189):
- a 3-channel input convolutional layer (convolution2D), with 5x5 kernel of depth 24, stride (2,2), relu activation and normatou initialization
- a hidden convolutional layer, with 5x5 kernel of depth 36, stride (2,2), relu activation and normal initialization
- a hidden convolutional layer, with 5x5 kernel of depth 48, stride (2,2), relu activation and normal initialization
- a hidden convolutional layer, with 3x3 kernel of depth 64, stride (1,1), relu activation and normal initialization
- a hidden convolutional layer, with 3x3 kernel of depth 64, stride (1,1), relu activation and normal initialization

The features extracted from the CNN is flattened, added drop-out, then applied to a 3 fully connected layers:
- layer 1 with output dimension 100, relu activation, normal initialization, and L2-regularizers with weight 0.001
- layer 2 with output dimension 50, relu activation, normal initialization, and L2-regularizers with weight 0.005
- layer 3 with output dimension 10, relu activation, normal initialization, and L2-regularizers with weight 0.01

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized using a custom normalization function (model.py lines 146-151)

As this is a regression problem, the loss was defined as the mean squared error between the predicted and recorded steering angle is used as the main optimization objective.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 184). 

L2-Regularizers were added in each of the full-connected layers.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, the learning rate was tuned manually (model.py line 192). Finally, I settled on a learning rate of 0.0001, as a balance between convergence speed and accuracy.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of images from the center, left and right cameras, with a heuristic weighing to exclude selective data.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

<video src="./res/run1.mp4" controls preload></video>
