from mytools import *

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import initializations
from keras import optimizers
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

# file locations
drivinglogdir = 'data'
logfilename = 'driving_log.csv'
drivinglogfile = os.path.join(drivinglogdir, logfilename)

############# load driving log data ############
def load_training_data(drivinglogfile):
    df=pd.read_csv(drivinglogfile, sep=',',header=0)

    #print(df.values[0,:])
    nSamples = df.shape[0]
    img_center = df.values[:,0]
    img_left = df.values[:,1]
    img_right = df.values[:,2]
    steering = df.values[:,3]
    throttle = df.values[:,4]
    brake = df.values[:,5]
    speed = df.values[:,6]

    X_center = []
    X_left = []
    X_right = []
    for i in range(nSamples):
        str_tmp = drivinglogdir + '/IMG/' + os.path.basename(img_center[i])
        X_center.append(mpimg.imread(str_tmp))
        str_tmp = drivinglogdir + '/IMG/' + os.path.basename(img_left[i][1:])
        X_left.append(mpimg.imread(str_tmp)) #remove first 'whitespace' character
        str_tmp = drivinglogdir + '/IMG/' + os.path.basename(img_right[i][1:])
        X_right.append(mpimg.imread(str_tmp)) #remove first 'whitespace' character

    return nSamples, X_center, X_left, X_right, steering, throttle, brake, speed

nSamples, X_center, X_left, X_right, steering, throttle, brake, speed \
    = load_training_data(drivinglogfile)
print('loaded ', nSamples, ' training samples with size ', X_center[0].shape)

# check data values by plotting
show_driving_data(steering, throttle, brake, speed, X_center, X_left, X_right, startframe=0, endframe=-1)

############# select and preprocess training data ############
def bin_steering(bindata):
    bin_res = []
    nLeft = 0
    nCenter = 0
    nRight = 0
    for i in range(len(bindata)):
        if bindata[i] < -0.05:
            bin_res.append(0)
            nLeft += 1
        elif bindata[i] > 0.05:
            bin_res.append(2)
            nRight += 1
        else:
            bin_res.append(1)
            nCenter += 1
    bin_stats = [nLeft, nCenter, nRight]
    #plt.hist(bin_res)
    #plt.show()
    return np.asarray(bin_res), np.asarray(bin_stats)

def index_ramp_end():
    i=0
    while speed[i]<30:
        i = i+1
    return i

# Assemble Training data
def assemble_training_data():
    bin_lcr, stats_lcr = bin_steering(steering)
    print('nLeft = ', stats_lcr[0], ', nCenter = ', stats_lcr[1], ', nRight = ', stats_lcr[2])
    #weightsteer = stats_lcr[np.argmin(stats_lcr)] * np.reciprocal(stats_lcr.astype(float))
    weightsteer = np.asarray([1.0, 0.5, 1.0])
    print('weightL = ',weightsteer[0], ', weightC = ',weightsteer[1], ', weightR = ', weightsteer[2])
    weight_lcr = []
    for i in range(len(steering)):
        weight_lcr.append(weightsteer[bin_lcr[i]].astype(float))

    coinflip = np.random.uniform(0.0,1.0,len(steering))
    keepthrow = coinflip <= weight_lcr

    # discard initial ramp-up
    startframe = index_ramp_end()
    print('Discard initial ramp-up, start from frame #', startframe )

    X_tmp = [] #np.array([]) #.reshape(X_center[0].shape)
    y_tmp = [] #np.array([]) #.reshape(steering[0].shape)
    for i in range(startframe, len(X_center), 1):
        if (keepthrow[i] == True):
            # center camera
            X_tmp.append(X_center[i])
            y_tmp.append(steering[i])

            #if(abs(steering[i])>0.05):
            if(steering[i]>0.05):
                # left camera
                X_tmp.append(X_left[i])
                y_tmp.append(steering[i]+0.2)

            if(steering[i]<-0.05):
                # right camera
                X_tmp.append(X_right[i])
                y_tmp.append(steering[i]-0.2)

    X_np = np.asarray(X_tmp)
    y_np = np.asarray(y_tmp)

    return X_np, y_np

X_np, y_np = assemble_training_data()
bin_lcr, stats_lcr = bin_steering(y_np)
print('Assembled ', X_np.shape[0], ' training data, with LCR: ', stats_lcr)

def crop_image(image_data):
    cutaway_top = 50
    cutaway_bottom = 20
    return(image_data[:,cutaway_top:-1-cutaway_bottom,:,:])

X_crop = crop_image(X_np)
crop_dim_1 = X_crop.shape[1]
crop_dim_2 = X_crop.shape[2]
print('cropped ', X_crop.shape[0], ' images to ',crop_dim_1, 'x', crop_dim_2)

def normalize_image(image_data):
    a = -0.5
    b = 0.5
    x_min = 0
    x_max = 255
    return a + ( ( (image_data - x_min)*(b - a) )/( x_max - x_min ) )

X_normalized = normalize_image(X_crop)

testframe = 4000
plt.figure(figsize=(10,15))
plt.subplot(3, 1, 1)
plt.imshow(X_np[testframe])
plt.subplot(3, 1, 2)
plt.imshow(X_crop[testframe])
plt.subplot(3, 1, 3)
plt.imshow(X_normalized[testframe])
plt.show()

# 1. For fit(): shuffle data
#X_train, y_train = shuffle(X_normalized, y_np)
# 2. For fit_generator(): shuffle and split validation set
X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_np, test_size=0.2, random_state=42)
############# create CNN in Keras ############

def create_model():
    # create baseline model (from keras lab)
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, init='normal',activation='relu',subsample=(2,2),\
        input_shape=(crop_dim_1, crop_dim_2, 3)))
    model.add(Convolution2D(36, 5, 5, init='normal',activation='relu',subsample=(2,2)))
    model.add(Convolution2D(48, 5, 5, init='normal',activation='relu',subsample=(2,2)))
    model.add(Convolution2D(64, 3, 3, init='normal',activation='relu',subsample=(1,1)))
    model.add(Convolution2D(64, 3, 3, init='normal',activation='relu',subsample=(1,1)))
    #model.add(MaxPooling2D((3, 3)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(100, init='normal', activation='relu', W_regularizer=l2(0.001)))
    model.add(Dense(50, init='normal', activation='relu', W_regularizer=l2(0.005)))
    model.add(Dense(10, init='normal', activation='relu', W_regularizer=l2(0.01)))
    model.add(Dense(1))

    # adjust learning rate
    adam = optimizers.Adam(lr=0.0001)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

model = create_model()

############# train model ############
train_datagen = ImageDataGenerator() #horizontal_flip=True)
train_generator = train_datagen.flow(X_train, y_train, batch_size=128, shuffle=True)
#samples_per_epoch=X_train.shape[0]
history = model.fit_generator(train_generator, samples_per_epoch=X_train.shape[0], nb_epoch=50, validation_data=(X_val, y_val))
#history = model.fit(X_train, y_train, batch_size=10, nb_epoch=10, shuffle=True, validation_split=0.2)

############# train and save model ############

# Save model to json and weights to file
def save_model(model):
    # save model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # save weights to HDF5
    model.save('model.h5')

model.summary()
save_model(model)
