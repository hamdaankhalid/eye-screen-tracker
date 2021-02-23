# Eye tracking on Screen using Convolutional Neural Network


## Goal & Overview
- The inspiration of this project came from an idea of how cool it would be if UX designers could track where on a webpage did user's look when given a task. I knew the problem could be divided into 
computer vision + data collection + and some sort of mapping that related eye and head movement to a screen. 

- Failed Approach: The inital approach was to do this with just  computer vision, by tracking eye balls
with some sort of depth perception from a screen and creating a function that maps each movement around the center of the screen. This approach was a grand failure as I learned the camera on my laptop was not sophisticated enough to deep dive into the pupils movements (this can be solved with a camera and lighting that is good enough), I also ran into errors about caliberating a user's distance as it was pretty had to recalibrate everytime there was achange in head position.

- Success(ish) Approach: The next approach involved into altering my problem statement from finding (x,y) coordinates on a screen, to estimates inside a region instead (this was more doable with my laptop cam). Now this became an image classification problem with the screen divided in a 3 by 3 matrix format with each cell named sequentially from 1-9. 


## Data Collection
- Regardless of the model I was going to use, I knew that I had to collect, clean, prepare train and test data( :sadtrombone ). For data collection I used opencv in a script that tracks your face and you are supposed to press the number related to the where the camera window is positioned when you are staring at it(e.g press 1 when looking at the camera is at tope left, then change position and keep doing so), the data collection script also created a training and testing split of 80% for training and 20% for testing data.


## What is a CNN and Why I used it
- After the Data collection was done I decided to use a CNN to classify these images, why? well, mostly because I made the assumption that CNN would do a better job at feature extraction because when i searched "best Models for image classification" CNN was everywhere.

For those not familiar with a CNN. CNN stands for Convolutional Neural Network. A Convolution is a mathematical operation on an image in it's 2d matrix format given a kernel or feature detector that produces a matrix called the feature map. A perfect example would be Instagram's filter image feature that lets you expose different aspects about an image you are uploading, e.g. some filters make the outlines of your image visible, some make the colors pop out more. Neural Network in my understanding is a blacked out box which takes an input, and does a computation that produces an equation with weights that corresponds to an output, it then checks if the output was closer to the desired output or further away, based on that on the next iteration it readjusts the weights, till it has some approximation that makes it's loss function happy (ofcourse this is a super vague explanantion but it provides a good overall understanding of the iterative process). A CNN combines the concept of Neural Networks and convolutions and takes a 2d matrix, perform multiple convolutions, and adjusts weights according to training data.


## Creating my super cute CNN
- I used kaggle as a learning tool to understand image classification and neural networks. I used transfer learning to decrease the amount of data I would need. By using [Resnet50](https://www.mathworks.com/help/deeplearning/ref/resnet50.html;jsessionid=fed1b28d7a40381f61327be4c0a9) and removing it's last layer (the prediction layer) I had a 50 layered model that I could tune for my purpose, I instead gave it a new prediction layer at the end with 9 nodes to classify the final result into (9 correspods to the  9 cells in our 3 by 3 matrix). The structure of the model is such that the convolutional layers(the ones which filter) are loaded by resnet's weights, and we only add a dense/full connected layer that inearly maps convolutional network's outputs to our dense layer for prediction. The activation function was set to softmax to return a probability, since I wanted to identify the n most likely places the user was looking at, and not a exact box which would have required the activation function to be ReLu. The model uses Stochaistic Gradient Descent to find the right weights for our model, and the metric for the model's success was accuracy (how many images were classified correctly). For our loss function we use Categorical cross entropy, so each predicted class probability is compared to the actual class desired output 0 or 1 and a score/loss is calculated that penalizes the probability based on how far it is form the actual expected value. The penalty is logarithmic in nature yielding a large score for large differences close to 1 and small score for small differences tending to 0.
Training the model, and testing it brought my attention to the fact that as the number of epochs increased, more number of times the weight are changed in the neural network and the curve goes from underfitting to optimal to overfitting curve. I initially used early stop, but then I decided to use the idea that as long as the training_accuracy < testing_accuracy I was safe from the fears of overfitting (for the sake of simplicity). I then made a live test where we have 9 boxes on our screen and our model gives predictions on where I may be looking on the screen in realtime.

<hr>

* Note: Although this project was initially meant to be tracking with the sole use of eye movement, head movement plays an even major part.
* It should also be noted that the model should be trained everytime there is a change in lighting, and position of monitor or webcam.
* the model works best on screens that are big. For my demonstration, I will be using an Acer 27"" Monitor as my screen and macbook pro's camera.

## Setup & Running
<ol>
<li> install dependencies as the errors show up 
<li> run data collector and take images
<li> run create_model notebook with the appropriate batch and epoch size till you have a model with good accuracy but with training accuracy < validation accuracy.
<li> run live_test.py to watch the model's preditions in realtime
  
  <hr>
