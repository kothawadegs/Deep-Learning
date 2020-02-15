# MNIST DATASET

The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset. It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9

## Constraints
### This code written under following constraints 
<p>1)99.4% validation accuracy
<p>2)Less than 20k Parameters
<p>3)Less than 20 Epochs
<p>4)No fully connected layer

## Requirements
<p>python 3.6
<p>pytorch 1.4.0

## Model
1)"In the following model I have included **Dropout** and **Batch_Normlization** to make the model more robust and prune to overfitting, BatchNormalization also takes care of diffrent intensities of images just in  case some imges can taken in dark or some at extreme daylight intensities.
<p>2)Maxpooling layer removes the 75% pixel values in image, it is done for computational efficiency.
<p>3)Also to avoid number of parameters within 20000 I have apply GAP ( Global average pooling) layer which just takes the mean of all the channels.ex- 6x6x10 results into 1x1x10 now we can apply fully connected layer after that.suppose 6x6x10 connected to 512 dense layer than number of parameters to tune become 6x6x10x512 = 1,84,320 but with GAP it bcomes 1x1x10x512 = 5120.
  <p>It is used for computational efficiency.
