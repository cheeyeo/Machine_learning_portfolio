## Promises of DL for CV

### Automatic Feature Extraction

Features can be automatically learnt and extracted from raw image data

Features provide context for inference about an image

CV uses hand-designed features such as SIFT, HOG etc; with DL, more complex and useful features can be learnt and extracted
from large image datasets through the use of deep NN, without the need for hand-engineered features

### End-to-end models

CV uses pipeline comprising of models which perform a specific task
e.g. in facial recognition, one model detects the faces in images; that's passed to another model in pipeline to recognize facial features

This pipeline approach can and is still used with deep learning models, where a feature detector model can be replaced with a deep neural network.

Alternately, deep neural networks allow a single model to subsume two or more traditional models, such as feature extraction and classification. It is common to use a single model trained directly on raw pixel values for image classification, and there has been a trend toward replacing pipelines that use a deep neural network model where a single model is trained end-to-end directly.

A good example of this is in object detection and face recognition where initially superior performance was achieved using a deep convolutional neural network for feature extraction only, where more recently, end-to-end models are trained directly using multiple-output models (e.g. class and bounding boxes) and/or new loss functions (e.g. contrastive or triplet loss functions).

### Model Reuse

Transfer learning

Use models trained on large image datasets and use that as a starting point on new projects i.e. remove the last layer of new NN and the weights feeding into last layer; retrain new NN on new training dataset

Pretrained models can be used to extract useful general features from digital images and can also be fine-tuned, tailored to the specifics of the new task

### Superior Performance

Performance improved year-on-year on range of computer vision tasks

Performance has been so dramatic that tasks previously thought not easily addressable by computers and used as CAPTCHA to prevent spam (such as predicting whether a photo is of a dog or cat) are effectively “solved” and models on problems such as face recognition achieve better-than-human performance.

### General Method

Top learning models developed from same set of components

i.e. Convolutional Neural Networks
     designed for image data and trained on image pixel data directly

A general class of CNN can be configured and used across each CV task directly


## Models used in CV

CNN specifically designed to work with image data

Multilayer Perceptrons (MLPs) used as inference model to make predictions based on features extracted by CNNs

RNN, esp LSTM, useful for working with image sequences over time such as videos


## Types of CV problems

Most DL for CV is used for object recognition or detection:

* Which object present in image

* Annotating an image with bounding boxes around each object

* Transcribe sequence of symbols from image i.e. character/text recognition

* Labeling each pixel of an image with identity of object it belongs to


