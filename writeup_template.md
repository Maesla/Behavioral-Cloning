#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia]: ./examples/nvidia_vehicle_control.png "Nvidia Architecture"
[model_mean]: ./examples/model_mean.png "Nvidia Architecture"
[driving_center]: ./examples/driving_center.jpg "Driving Center"
[driving_to_the_center]: ./examples/driving_to_center.jpg "Driving to center"
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I am using the NVIDIA architecture explained in Lesson 12, Point 14. Even More Powerful Network.

![alt text][nvidia]


The architecture is pretty much the same. I have made these changes:

1. The input shape. In my case is 160,320,3
2. I have added two layers at the beginning. 
 3. One first layer for normalizing the values. **model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))**
 4. A second layer for cropping the not interesting areas. **model.add(Cropping2D(cropping=((50,20), (0,0))))**
5. I have added a fully connected layer at the end, with only one neuron (the steering wheel)

The whole model can be found in [model.py](./model.py), line 86
####2. Attempts to reduce overfitting in the model

The model contains dropout layers after all the layers in order to reduce overfitting [model.py](./model.py)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). I split the data in three sets. 

- Training set: 0.8
- Validation set: 0.1
- Evaluation set: 0.1

As noted in the final training epoch, validation loss y evaluation loss is very similar, so overfitting should not be a problem. After 10 epochs, validation loss = 0.0351 and evaluation loss =  0.03812

![alt text][model_mean]


21048/21048 [==============================] - 115s - loss: 0.0594 - val_loss: 0.0453

Epoch 2/10
21048/21048 [==============================] - 103s - loss: 0.0400 - val_loss: 0.0405

Epoch 3/10
21048/21048 [==============================] - 107s - loss: 0.0376 - val_loss: 0.0376

Epoch 4/10
21048/21048 [==============================] - 98s - loss: 0.0361 - val_loss: 0.0379

Epoch 5/10
21048/21048 [==============================] - 107s - loss: 0.0360 - val_loss: 0.0373

Epoch 6/10
21048/21048 [==============================] - 105s - loss: 0.0353 - val_loss: 0.0372

Epoch 7/10
21048/21048 [==============================] - 126s - loss: 0.0349 - val_loss: 0.0364

Epoch 8/10
21048/21048 [==============================] - 109s - loss: 0.0343 - val_loss: 0.0352

Epoch 9/10
21048/21048 [==============================] - 100s - loss: 0.0334 - val_loss: 0.0333

Epoch 10/10
21048/21048 [==============================] - 108s - loss: 0.0333 - val_loss: 0.0351

Evaluation. Samples 439
loss: 0.038127114375432335


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).

####3. Creation of the Training Set & Training Process

To capture data, I recorded to laps.

- The first lap is a standard lap, trying to drive always at the lane center.
- In the second lap, I tried to provide data about what to do if the vehicle goes out of the lane.

*Driving at the center*

![alt text][driving_center]

*Driving to the center*

![alt text][driving_to_the_center]


After the collection process, I had 4385 number of data points.
The final set was increased in two ways.

1. First, I used the 3 cameras. I added the three images (left, center, right) to the set. About the steering angle, I used a correction factor. So the angle = angle +Q*correction_factor. This can be found at [model.py](./model.py) (line 56)
  2. Q = 0 for the center image
  3. Q = 1 for the left image
  4. Q = -1 for the right image
5. The second way to increase the data set was the mirroring approach. All the set is mirrored, and the angle is mirrored too. This can be found at [model.py](./model.py) (line 68)

So, the final set is 4385 data points x 3 cameras x 2 (one normal, one mirrored) = 4385*3*2= 26310

I have found that a import problem could the input itself. The vehicle input has to be either the keyboard or dragging the mouse. Although the keyboard has been setup with sensibility (the input is not only 0 or 1), the Unity Wheel Collider is very sensitive to hard turn. I have the feeling that with a real steering wheel, like a logitech G29 or even with a video game controller the data acquisition could be quite better.



