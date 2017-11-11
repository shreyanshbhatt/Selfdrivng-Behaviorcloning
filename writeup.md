#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model using NVDIA convnet architecture
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained NVDIA convolution neural network
* model_overfit.h5 containing a trained NVDIA convlution network with 2 laps data
* writeup_report.md

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. Model Architecture and Training Strategy

I tried with the regression model first as started with the project introduction. I tried tried my LeNet model I used for project 2.
The issue with that model was it got stuck in the local minima. It outputs 25 degree angle making the car circle in clockwise direciton. Also, the training and validation loss was really high (>10.0). I noticed the issue of underfitting and I removed the dropout layers I had between my full connected layers of LeNet architecture.

It improved the validation loss however it was not enough to drive the car for turns.

Next, I chose to use the NVDIA convnet architecture since it has been popular for driving. The pipeline consists of following,

Normalization - subtracting 127.5 and mean centering
Image cropping
Convolution2d - 24 - 5x5 filters with 2x2 strides, ReLU activation
Convolution2d - 36 - 5x5 filters with 2x2 strides, ReLU activation
Convolution2d - 48 - 5x5 filters with 2x2 strides, ReLU activation
Convolution2d - 64 - 3x3 filters with 1x1 strides, ReLU activation
Convolution2d - 64 - 3x3 filters with 1x1 strides, ReLU activation
Fullyconnected lyaer
Fullyconnected layer
Fullyconnected layer
Fullyconnected layer - value

####2. Attempts to reduce underfitting/overfitting in the model
I started the project with an issue of underfitting for my traffic sigh image classification LeNet network. I tried the same architecture without dropout layers which performed well.
I considered flipped image with inverted angle and considering left and right camera images with correction value 0.1 as suggested in the project introduction which also helped reducing training and validation loss. However, the data I started with was a very 100 images data which included images captured by driving only a couple of seconds on track.
I noticed an issue of overfitting when I considered 1 lap data that I collected by driving the car for 1 lap. Next, I generated data for 2 laps and with the other lap having data for how to recover after drifting too right or too left. I noticed an overfitting on this dataset also the car did not make all the sharp turns.
Next, Motivated by the Transfer learning, I used the same trained model (with NVDIA convnet trained on 2 laps dataset I mentioned) and trained it on sample data provided with the project. It still had an overfitting issue. I suspected that being a reason of training a model with large data which was previously trained for a small data. (Not entirely sure about it and still investigating).
Next, I did the training process in reverse that I trained the model with sample data provided with the project (the car still did not make one sharp left turn near water with only this) and then trained it with the 2 laps data I collected. I noticed the validation loss and training loss being almost equal and ~ 0.009

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Training data selection
Point 2 partially addresses this.

I used the flipped image, right camera, and left camera with correction in steering angle.
I collected 2 laps data with 1 lap where I drifted and recovered car.
I used the sample training data provided with the project.

####5. Model behavior

After the collection process, I had ~7K images from what I collected data of 2 laps and ~24k images from sample training data. I used the inverted images with inverted steering angle which resulted in ~10k data points and ~40k data points resp. I had 80%-20% training validation split.

I noticed the model overfitting for the scenarios I described above. The model trained with 2 laps dataset followed by sample dataset could drive the car but was not able to recover when drifted car manually. (overfit_model_with_drifts.mp4)

On the other hand, the model trained with sample training data first followed by the 2 laps training data was robust enough to manage manual drifts in between. (run_with_manual_drifts.mp4)
