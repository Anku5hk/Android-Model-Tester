# About
This Android application is used to test inferences of Tensorflow/Pytorch graphs. This application supports variety of Models, Computer Vision tasks independent of library. Also one can use it to learn how the code works for pytorch and tensorflow. Soon more tasks will be added gradually.

# Features
- Fast inference on android for tensorflow/pytorch graphs.
- Supports Classification, Object Detection.
- Simple GUI based options for inputs.
- Auto detect library(Tensorflow/Pytorch).
- Small size and easy to understand code(Tons of comments).
- Open source: Edit/Make changes, use as a starter template No Worries.

# How To?
1. Train your model.
2. Save it as .pt for pytorch or .tflite for tensorflow.
3. Put labels(classes names) into a simple .txt file, no formatting needed only newline for new class. 
   Same for object detection.
4. Select files through application setting.
5. Select image and press button.
6. Done.

# Screens
![3](/screenshots/sss4.PNG)
![1](/screenshots/sss1.PNG)
![2](/screenshots/sss2.PNG)
![1](/screenshots/sss3.PNG)

# More Features to be added.
- More Image tasks.

# More
You can compile above code into Android studio or download apk [here](https://drive.google.com/open?id=1qn0yiFxyEcxa4EVHbDeL4mErxkysXgdS)<br/>
Classification Models Zoo: <br/>
(https://github.com/qubvel/classification_models)<br/>
Object Detection Models Zoo: (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

# Uppdate
- Changed code to completely follow the official documentations.
- Added support for camerax.
- Tensorflow runs on 0.0.0-nightly version.
- Supports Quant/Float models on Tensorflow.
