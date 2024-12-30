# SignLanguageRecognition
# A Mini Project Synopsis on 
Sign Language Recognition using Convolutional Neural Networks in Machine Learning

# INTRODUCTION
Sign Language is one the basic means of communication used by physically impaired individuals. Most people cannot understand the sign language used by these individuals and there is a need for a user friendly and efficient real time translation of the hand gestures made by the disabled person into natural language such as English in the form of text or sentences. In this project the technique of machine learning and its method of convolutional neural networks is used for image processing and conversion.

The proposed software or program is created using Python as the basic programming language and its supported libraries which include Keras, Tensorflow, Numpy and OpenCV. The project consists of three parts: Creating the dataset for training, training the CNN on the data captured and predicting the data in real-time. 

In this project the dataset will be created by using the video camera live feed where each frame of the video camera will attempt to detect the region of interest of the hand. Each frame which captures region of interest is saved in the directory. Two folders called train and test each having ten folders containing images captured using the video camera. Here the live camera feed is obtained using OpenCV which will detect the region of interest of the hand seen on live feed. The hand will be identified on live feed inside a red box shown on the screen. The background is differentiated by calculating the accumulated weighted average for background and then subtracted from frames that contain some object in front of the background which can be distinguished as foreground.

The threshold value is calculated for every frame to determine the contours using OpenCV and the maximum contours are returned suing function segment. The contours help to determine if there is any foreground object in the feed. When contour is detected the image is saved in the train and test set for the letter or number we are detecting. We save over 500 images of each number and alphabet to be detected for the train data and over 50 images for test data.

Now the training of the CNN is performed by loading the data using ImageDataGenerator of the keras library, and the images are plotted using the plotImages function. The CNN is designed based on some trial-and-error parameters. The model is then fitted and saved for use in the last module. In training, call-backs of Reduce LR on plateau and early stopping is used. After every epoch the accuracy and loss is calculated using validation dataset and if the loss is not decreasing the LR of the model is reduced using Reduce LR to prevent model from overshooting minima of loss and early stopping algorithm is used for stopping training if accuracy keeps on decreasing. An optimization algorithm called Stochastic gradient descent (SGD) is used for accuracy purposes. After compilation of model, the model is fit on train batches of 10 epochs using call backs. We now get the next batch of images from test data and evaluate model on test set and print accuracy and loss scores.

Finally, the bounding box for detecting the region of interest is created and accumulated average is calculated by identifying foreground object as we had done during dataset creation. Then we find the maximum contour and if contour is detected this means that hand is detected and hence threshold of the region of interest is treated as a test image. The previously saved model is loaded and the fed with the threshold of image of region of interest containing the hand as input to the model for prediction. The model created earlier is loaded and set with some variables such as initializing background variables and setting the dimensions of the region of interest. The function to calculate background accumulated weighted average is used and the hand is segmented by getting maximum contours and threshold image of hand detected.

# OBJECTIVES
•To help the dumb and deaf to optimise their communication with others.

•Creating a sign detector, which detects numbers from one to ten

# SYSTEM REQUIREMENTS
Software Requirements: Python, Jupyter IDE, Numpy, CV2(open cv), Keras and Tensorflow.

Hardware Requirements: RAM-8GB, Processor-Intel I5, 64-bit Operating System, x64 based processor.	
 
# EXPECTED IMPLEMENTATION METHOD
Implementation involves:

1)Creating a Data Set

2)Training a CNN on the captured Data set

3)Predicting the data

# Creating the Data set

•We will be creating a data set and getting a live feed from the video cam which is saves in a directory.

•Now for creating the dataset we get the live cam feed using OpenCV and create an ROI.

•We will have an accumulated weighted average in order to distinguish the object as foreground.

•After we have the accumulated average for the background, we subtract it from every frame that we read after 60 frames to find any object that covers the background.

•We calculate the threshold value and determine if there is any foreground object being detected.

# Training the CNN on the Captured data set

•We load the data using ImageDataGenerator of keras through which we can use the flow_from_directory function to load the train and test set data.

•plotImages function is for plotting images of the dataset loaded.

•We fit the model and save the model for it to be used.

# Predicting the data

•We create a bounding box for detecting the ROI and calculate the accumulated average and identify a foreground object.

•We find the max contour and if contour is detected that means a hand is detected so the threshold of the ROI is treated as a test image.

•We load the previously saved model using keras.models.load_model and feed the threshold image of the ROI consisting of the hand as an input to the model for prediction.

•Detecting the hand now on the live cam feed.

# EXPECTED NEW APPLICATIONS
•Conversion of sign language to English.

# CONCLUSION	
Sign Language Recognition System has been developed from classifying static signs and alphabets, to the system for recognition. After the recognition of the static signs, we will try to implement dynamic signs for recognition. This project will help the dumb and deaf people to communicate with others in an efficient manner. This project is mainly focused on static signs/ manual signs/ alphabets/ numerals.
