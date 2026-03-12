# Emotion Classification Model
CompSci 4452 Deep Learning in Computer Vision - Project \
Liam Truss \
April 3rd, 2026

## Dataset
The dataset used is the FER 2013 dataset containing 7 emotions (anger, disgust, fear, happiness, neutral, saddness and surprised) \
Dataset link: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

## Description
This project investigates convolutional neural network–based facial emotion recognition using grayscale images. A baseline CNN is compared against an attention-augmented CNN to evaluate whether spatial and channel attention improves classification performance on a dataset. The models are analyzed using quantitative metrics (accuracy, F1-score, confusion matrices) and interpretability techniques such as Grad-CAM visualization to identify if the attention model encourages the model to focus on semantically relevant facial regions (i.e. brow position and mouth shape).  

## Typical pipeline
1. Run 'train.py' to generate models.
2. Run 'evaluate.py' to generate confusion matricies and t-SNE graphs.
3. Execute 'predict.py' to predict the emotions of the images in the 'Images' directory, also generating Grad-CAM heatmaps outputted to 'GradCAM/'.

## Files and Directories
Models - Pre-trained models for the baseline and attention models \
TestData - All data used for validating models \
TrainData - All data used for training models \
dataset.py  - Handles dataset loading, preprocessing, and data-augmentation for training models \
evaluate.py - Evaluates trained models using confusion matrices and t-SNE clustering \
model.py - Defines the baseline CNN and attention-augmented CNN architectures for emotion classification \
predict.py - Runs inference on input images and generates Grad-CAM visualizations on top of images \
train.py - Trains and validates the models, tracking accuracy, F1-score, and saving the best-performing model (largest F1 score)