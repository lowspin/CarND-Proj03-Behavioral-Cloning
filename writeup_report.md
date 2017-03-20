# **Behavioral Cloning Project** 

## Background

This repository contains my project report for the [Udacity Self-Driving Car nano-degree](https://www.udacity.com/drive) program's project 3 - Behavioral Cloning. The original starting files and instructions can be found [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3.git). Sample training [data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by udacity were used, along with udacity's [driving simulator](https://github.com/udacity/self-driving-car-sim). The project was done in Ubuntu 16.04 LTS using local Nvidia GTX960M GPU.

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
[image2]: ./res/driving-data-stats.png "Visualize udacity's driving data"
[image3]: ./res/sterring_histogram.png "steering angle histogram"
[image4]: ./res/left_center_right_camera_images.png "Sample left, center and right camera images"
[image5]: ./res/example_pre-processing.png "Sample pre-processing for each frame"
[image6]: ./res/Experiment_temporal_Info.png "Experimental dual frame processing"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py - main script to create and train the model
* mytools.py - containing miscellaneous visualization functions
* drive.py - for driving the car in autonomous mode
* model.h5, model.json - containing a trained convolution neural network 
* writeup_report.md (this file) - summarizing the results
* /res/run1.mp4 - video generated from autonomous driving around track 1 (in case evaluation fails).
* model2.py, drive2.py, model2.h5, model2.json - experimental dual-frame mode (see below)

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. Keras's ImageDataGenerator was used with fit_generator to batch-generate training data rather than storing the training data in memory. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

My model is based on the original Nvidia paper ["End to End Learning for Self-Driving Cars"](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). Drop-out and L2-regularization were added to prevent over-fitting. (Please see details about the final model architecture in the next section below).

As this is a regression problem, the loss was defined as the mean squared error between the predicted and recorded steering angle is used as the main optimization objective.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 184). 

L2-Regularizers were added in each of the full-connected layers.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, the initial learning rate was tuned manually (model.py line 192). 

Batch size is set with the Keras ImageDataGenerator and is set by trial and error. (model.py line 202)

Similarly, number of Epochs set in Keras model.fit_generator is set by trial and error, the entire training set was used in each Epoch. (model.py line 204)

Finally, as a balance between convergence speed and accuracy, I settled on:
- initial learning rate of 0.0001, 
- batch size of 128,
- 50 Epochs

#### 4. Appropriate training data

Initially, I collected my own training data (attached [training01.zip](training01.zip)) using the Udacity simulator on track 1. However, I found that the track has very few right bends/turns. As a result, I have to drive many round to have sufficient positive steering angle samples. I could flip the negative steering images and invert the sign of the steering angle to get more positive value samples, or drive the circuit in reverse. In the end, I chose to use the Udacity provided driving data, which has a fairly even distribution between positive and negative steering samples, to train my network.

Training data was chosen to keep the vehicle driving on the road. I used a combination of images from the center, left and right cameras, with a heuristic weighing to exclude selective data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with an established working model and tweak hyper-parameters. Emphasis is placed on training data selection to achieve required accuracy.

My first step was to use a convolution neural network model similar to the Nvidia paper ["End to End Learning for Self-Driving Cars"](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I thought this model might be appropriate because is was used for a autonomous driving application using only camera images, which is exactly what we're trying to do.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used the train_test_split function from sklearn.model_selection to perform a random split (model.py line 168).

To combat the overfitting, I modified the model as follows. 
- Drop-out with keep probability 0.5. This was applied after the convolution neural network, before the fully connected layers. (model.py line184) 
- L2-regularization is also added to all the weights in the fully connected layers, with varying weights depending on the depth of the layer. (model.py lines 186-188)

Then I fine-tune the intial learning rate of the ADAM optimizer, by manually adjusting a few values (0.01, 0.005, 0.001, 0.0005, 0.0001 etc) and observing the autonomous driving results (model.py line 192)

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially in sharp bends. To improve the driving behavior in these cases, I went back to the training data and select samples with a more intentionally designed criteria (see below).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes.

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

Here is the model summary of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Model Summary][image1]

#### 3. Creation of the Training Set & Training Process

The driving data provided three different camera images, captured from three locations on the windshield. Below is a sample of the left, center and right camera images.

![camera images][image4]

These are the plots of the steering angle, throttle/brake and resulting speed of the data.
![Driving data stats][image2]
Since, we're not adjusting throttle and speed in this project, it is important that the training data is also collected without major changes in throttle and speed. There is a small section where these changes more than usual, but overall, the data looks ok.

As we can see from the histogram of the steering angles, there are a lot more samples with straight road segments (steering angle close to 0)
![steering angle histogram][image3]

If we bin the steering angles into three sets (model.py lines 64-82):
- less than -0.05 (left steering)
- between -0.05 and +0.05 (straight)
- more than +0.05 (right steering)

We see that the dataset consists of 1532 left steering, 4881 straight and 1623 right steering. If we use these data without filtering, the model will mostly learn how to "drive" (adjust steering angle) on straight segments, without much experience in cruves and bends. To overcome this, I applied a weighting probability to the training images (model.py line 95). Specifically, since the left and right samples are about equal, I apply "select" probabilities of:
- left steering samples keep probability = 1.0
- straight steering samples keep probability = 0.5
- right steering samples keep probability = 1.0

Hence, half of the straight steering samples were dropped.

In addition, to help with sharp turns and bends, I selectively added left and right camera images as follows:
- if steering bin is left, add the right camera images, with additional -0.2 added to the original steering angle
- if steering bin is right, add the left camera images, with additional +0.2 added to the original steering angle
- if steering bin is straight, no side camera images were added.

Finally, as a pre-caution (from my own experience), I discard the initial period when the car is ramping up to cruising speed, as most of the time, we're reacting to images at constant speed with steering control only.

After the data selection, I have the following distribution, which seems appropriate for our training:
- left steering samples = 3062
- right steering samples = 3220
- approx zero steering samples = 2437

#### Data augmentation
I have chosen not to augment the dataset, as the training samples was sufficient. The Keras ImageDataGenerator I'm using can be used to generate additional samples with e.g. horizontal flips, random rotation, shifts and whitening, but it is not clear how to adjust the y labels accordingly using the generator.

#### Pre-processing
After the collection process, I had 8719 number of data points. I then preprocessed this data by:

Cropping (model.py lines 136-144):
In order to use the side camera images as training samples, I have cropped the bottom of the images to remove the "chasis" of the car, which are on different side of the images according to the camera source. In addition, I have also cropped out the top "sky" part of the images. This is both to save on processing and to help with environments the model have not seen before (e.g. track 2), as we ignore these background objects.

Normalization (model.py lines 146-153):
Simple linear normalization is performed to scale each RGB pixel to within -0.5 and +0.5.

The following is an example of an image in each of the pre-processing steps.
![Pre-processing][image5]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was chosen as 50 to balance between accuracy and speed.

### Simulation
After training, the car was able to navigate autonomously around the track without leaving the drivable track surface. (In case there are problems loading the model into the simulator, I have uploaded a video of autonomous driving around [track 1](/res/run1.mp4).

### Experiment - Dual Frame Processing
Since we shuffle the images random and train with them as individual samples, none of the temporal information is retained in the training data after assembly and pre-processing. For example, if the car moves from the center to the side of the lane in consecutive frame, the corresponding steering maybe to correct for unintentional swerving, but if the car has been consistently staying one side for consecutive frames and steering is postive or negative, the human driver could be "hugging the bend" while negotiating a curve. 

**Data Assembly**
To this end, I conducted an experiment to see if we can do better by using some information from the previous frame in the training. My original intention was to expand the input convolution layer to 6 channels - the first 3 layers are the current frame's RGB signal, while the next 3 were the previous frame's. All 6 layers are passed into the CNN and go through similar processing as in the 3 channel case. For a 10Hz simulator, this would add a bit of temporal information from 100ms ago. 

However, I ran into two problems:
1. The Keras convolution2D layer does not seem to like 6 channels as input - only 1,3,4 were possible when I tried. I did not not have time to try other architecture.
2. My machine ran out of memory very quickly as large 6 channels numpy arrays needed to be created.

To workaround, I convert the previous frame's data into (single channel) grayscale, and use a 4-layer input, i.e. 1st 3 layers are the current frame's RGB image, and the 4th layer is the previous frame's grayscale image.

![Experimental dual frame mode][image6]

**Model adjustment**
Since input is increase by 4/3, I increase the depth of all layers by the ratio to accomodate the additional information flowing through. The resulting model is as follows:
- a *4-channel* input convolutional layer (convolution2D), with 5x5 kernel of depth *32*, stride (2,2), relu activation and normatou initialization
- a hidden convolutional layer, with 5x5 kernel of depth *48*, stride (2,2), relu activation and normal initialization
- a hidden convolutional layer, with 5x5 kernel of depth *64*, stride (2,2), relu activation and normal initialization
- a hidden convolutional layer, with 3x3 kernel of depth *86*, stride (1,1), relu activation and normal initialization
- a hidden convolutional layer, with 3x3 kernel of depth *86*, stride (1,1), relu activation and normal initialization
- drop-out with keep probability 0.5
- fully connected layer 1 with output dimension *134*, relu activation, normal initialization, and L2-regularizers with weight 0.001
- fully connected layer 2 with output dimension *67*, relu activation, normal initialization, and L2-regularizers with weight 0.005
- fully connected layer 3 with output dimension *14*, relu activation, normal initialization, and L2-regularizers with weight 0.01

**Files**
The corresponding files for the experiment are in this repo:
* model2.py - train and save model using the dual-frame format
* model2.h5, model2.json - final saved model
* drive2.py - added a few lines to save the previous frame, convert to grayscale and stack at the bottom of the 4-layer input.

To drive the car autonomously using udacity's simulator, execute
```sh
python drive2.py model2.h5
```

**Result**
Unfortunately, the current result is worse than single frame processing. The car is not yet able to navigate the first track. I think there could be a few reasons for this:
- the model is not optimal for this new input - more refinement is needed (maybe more layers?) than just merely scaling the number of weights,
- Just gray-scaling the previous frame is not an optimum way to extract temporal information, may motion vectors could be better,
- One frame is too short to have any meaningful change in the image,
- A recurrent network model is required, added between the convolutional network and fully connect layers.

### Reflection
The biggest lesson I learn from this project is that the deep neural network designed for any particular application is only as good as the data we feed into it. We can spend a lot of time fine-tuning the network, but if we do not select or pre-process the training data appropriately, the result is going to be inaccurate. 

The second lesson I learn from the experiment is that the deep neural network model should be adapted for the input data as well. In transfer learning, we cannot expect one network that worked on another problem to work seamlessly on our current problem, without any addition or changes to the original architecture.
